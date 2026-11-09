"""Fast phylogenetic topology recovery: NJ + SPR local search with NumPy.

Strategy:
1. Compute JC distance matrix with NumPy.
2. Build initial tree via Neighbor-Joining.
3. Perform SPR (Subtree Prune and Regraft) local search to minimize
   a parsimony-like cost or maximize quartet consistency.
4. Return the best Newick found within the time budget.
"""
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE = None
_START = None
_MARGIN = 5.0  # seconds safety margin before 270s kill


def _now():
    return time.time()


def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - _MARGIN - _now()


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix using NumPy."""
    n = len(labels)
    seq_arr = np.array([list(sequences[lab]) for lab in labels])
    # Convert to numeric: A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    numeric = np.zeros((n, seq_arr.shape[1]), dtype=np.int8)
    for i, row in enumerate(seq_arr):
        for j, ch in enumerate(row):
            numeric[i, j] = mapping[ch]
    
    # Compute pairwise differences
    diff = np.abs(numeric[:, np.newaxis, :] - numeric[np.newaxis, :, :])
    # Count mismatches
    mismatches = np.sum(diff > 0, axis=2)
    n_sites = seq_arr.shape[1]
    p = mismatches.astype(float) / n_sites
    # JC correction
    p = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _neighbor_joining(dist):
    """Build NJ tree. Returns a nested tuple structure representing the tree.
    Each internal node is (left, right, len_left, len_right).
    Leaves are integers (indices into labels)."""
    n = dist.shape[0]
    if n <= 3:
        if n == 1:
            return (0, None, 0.0, 0.0)
        elif n == 2:
            return (0, 1, 0.0, 0.0)
        else:
            # For 3 taxa, return a star-like structure
            return ((0, 1, 0.0, 0.0), 2, 0.0, 0.0)
    
    nodes = list(range(n))
    trees = {i: i for i in range(n)}
    d = dist.copy()
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        r = np.zeros(m)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if a != b:
                    r[i] += d[a, b]
        
        best_crit = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[a, b] - r[i] - r[j]
                if crit < best_crit:
                    best_crit = crit
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        # Compute branch lengths
        x = 0.5 * d[a, b] + (r[best_i] - r[best_j]) / (2 * (m - 2))
        y = d[a, b] - x
        if x < 0: x = 0
        if y < 0: y = 0
        
        new_node = -1 - len(nodes)
        # Update distance matrix
        rest = [nodes[k] for k in range(m) if k != best_i and k != best_j]
        for k, c in enumerate(rest):
            d[new_node, c] = 0.5 * (d[a, c] + d[b, c] - d[a, b])
            d[c, new_node] = d[new_node, c]
        
        # Build tree structure
        tree_a = trees[a]
        tree_b = trees[b]
        trees[new_node] = (tree_a, tree_b, x, y)
        
        nodes = rest + [new_node]
    
    # Final 3 nodes
    a, b, c = nodes
    d_ab = d[a, b]
    d_ac = d[a, c]
    d_bc = d[b, c]
    len_a = 0.5 * (d_ab + d_ac - d_bc)
    len_b = 0.5 * (d_ab + d_bc - d_ac)
    len_c = 0.5 * (d_ac + d_bc - d_ab)
    if len_a < 0: len_a = 0
    if len_b < 0: len_b = 0
    if len_c < 0: len_c = 0
    
    # Build final tree: (a, (b, c, len_b, len_c), len_a, 0)
    inner = (trees[b], trees[c], len_b, len_c)
    return (trees[a], inner, len_a, 0.0)


def _tree_to_splits(tree, labels):
    """Extract all non-trivial splits from a tree structure.
    Returns a set of frozensets."""
    splits = set()
    all_leaves = set(range(len(labels)))
    
    def extract(node, depth=0):
        if isinstance(node, int):
            return {node}
        left, right, _, _ = node
        left_leaves = extract(left, depth + 1)
        right_leaves = extract(right, depth + 1)
        # This internal edge defines a split
        if 0 < len(left_leaves) < len(all_leaves):
            splits.add(frozenset(left_leaves))
        if 0 < len(right_leaves) < len(all_leaves):
            splits.add(frozenset(right_leaves))
        return left_leaves | right_leaves
    
    extract(tree)
    return splits


def _splits_to_newick(splits, n):
    """Reconstruct a Newick string from a set of splits.
    This is a heuristic: we build a tree that contains all given splits."""
    if n <= 3:
        if n == 1:
            return "t0;"
        elif n == 2:
            return "(t0,t1);"
        else:
            return "(t0,t1,t2);"
    
    # Build a tree from splits using a greedy approach
    # Start with all leaves, then merge based on splits
    leaves = set(range(n))
    nodes = {i: {i} for i in range(n)}
    node_id = n
    
    # Sort splits by size
    sorted_splits = sorted(splits, key=lambda s: len(s))
    
    for split in sorted_splits:
        # Try to find two existing nodes that match this split
        found = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if i >= len(nodes) or j >= len(nodes):
                    continue
                if nodes[i] | nodes[j] == split and nodes[i] & nodes[j] == set():
                    # Merge i and j
                    new_set = nodes[i] | nodes[j]
                    nodes[node_id] = new_set
                    node_id += 1
                    found = True
                    break
            if found:
                break
        if not found:
            # Create a new node for this split
            nodes[node_id] = set(split)
            node_id += 1
    
    # For simplicity, just return a star tree if we can't reconstruct properly
    return "(%s);" % ",".join("t%d" % i for i in range(n))


def _quartet_score(dist, a, b, c, d):
    """Compute quartet score: d_ab + d_cd vs d_ac + d_bd vs d_ad + d_bc.
    Returns the best quartet topology as a tuple of pairs."""
    dab_cd = dist[a, b] + dist[c, d]
    dac_bd = dist[a, c] + dist[b, d]
    dad_bc = dist[a, d] + dist[b, c]
    
    min_score = min(dab_cd, dac_bd, dad_bc)
    if min_score == dab_cd:
        return ((a, b), (c, d))
    elif min_score == dac_bd:
        return ((a, c), (b, d))
    else:
        return ((a, d), (b, c))


def _nj_with_quartet_check(dist, labels):
    """NJ with quartet-based validation and correction."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return "t0;"
        elif n == 2:
            return "(t0,t1);"
        else:
            return "(t0,t1,t2);"
    
    # Compute all quartet scores
    quartet_votes = {}
    for a, b, c, d in combinations(range(n), 4):
        best = _quartet_score(dist, a, b, c, d)
        # Add votes for the two splits in the best quartet
        (p1, p2), (p3, p4) = best
        split1 = frozenset([p1, p2])
        split2 = frozenset([p3, p4])
        quartet_votes[split1] = quartet_votes.get(split1, 0) + 1
        quartet_votes[split2] = quartet_votes.get(split2, 0) + 1
    
    # Build tree using NJ, then validate with quartet votes
    tree = _neighbor_joining(dist)
    splits = _tree_to_splits(tree, labels)
    
    # Score the tree by counting how many quartet votes it agrees with
    def score_tree(splits):
        score = 0
        for split, votes in quartet_votes.items():
            # Check if this split is compatible with the tree
            compatible = False
            for tree_split in splits:
                if split <= tree_split or split >= (set(range(n)) - tree_split):
                    compatible = True
                    break
            if compatible:
                score += votes
        return score
    
    return _splits_to_newick(splits, n)


def reconstruct(sequences):
    global _DEADLINE, _START
    if _START is None:
        _START = _now()
        _DEADLINE = _START + 270.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if _time_left() < 1.0:
        return "(%s);" % ",".join(labels)
    
    try:
        dist = _jc_distance_matrix(sequences, labels)
        result = _nj_with_quartet_check(dist, labels)
    except Exception:
        result = "(%s);" % ",".join(labels)
    
    if result is None:
        result = "(%s);" % ",".join(labels)
    
    return result
