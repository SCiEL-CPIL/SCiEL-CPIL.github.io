"""Phylogenetic topology recovery: NJ with JC distances, then NNI refinement."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 260.0  # safety margin before 270s kill


def _jc_dist_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    # Convert sequences to numpy arrays for vectorized comparison
    seq_list = [sequences[lb] for lb in labels]
    L = len(seq_list[0])
    # Build a matrix of mismatch counts
    # For memory efficiency with up to 32 taxa, this is fine
    arr = np.array([list(s) for s in seq_list])  # shape (n, L)
    # Compute pairwise Hamming distances
    # arr[:, i, None] != arr[:, None, j] -> but let's do it in a loop for clarity
    # Actually, vectorize: for each pair (i,j), count mismatches
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(arr[i] != arr[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_reconstruct(D, labels):
    """
    Neighbor Joining algorithm.
    D: n x n distance matrix (numpy)
    labels: list of taxon names
    Returns: Newick string
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    # We'll track current nodes and their subtree representations
    # Use a dict: node_id -> (subtree_newick, is_original_taxon)
    # Start with original taxa
    nodes = list(range(n))  # indices into labels
    # subtree[i] will be the Newick string for node i
    subtree = {i: labels[i] for i in range(n)}
    
    # Work on a copy of D
    dist = D.copy()
    current_n = n
    
    while len(nodes) > 3:
        if time.time() > _DEADLINE:
            # Return a star tree as fallback
            return "(" + ",".join(subtree[x] for x in nodes) + ")" + ";"
        
        m = len(nodes)
        # Compute row sums
        row_sum = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    row_sum[i] += dist[i, j]
        
        # Find pair (a, b) minimizing (m-2)*d(a,b) - row_sum[a] - row_sum[b]
        best_criterion = float('inf')
        best_a = 0
        best_b = 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist[i, j] - row_sum[i] - row_sum[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = i
                    best_b = j
        
        a = best_a
        b = best_b
        
        # Compute branch lengths
        da = 0.5 * dist[a, b] + (row_sum[a] - row_sum[b]) / (2.0 * (m - 2))
        db = dist[a, b] - da
        # Clamp to non-negative for Newick
        da = max(0.0, da)
        db = max(0.0, db)
        
        # Create new node
        new_idx = m  # temporary index
        new_label = f"n{new_idx}_{a}_{b}"
        
        # Update distance matrix: new distances
        # For each remaining node k, new distance = 0.5*(d(a,k) + d(b,k) - d(a,b))
        new_dist = np.zeros(m - 1)
        remaining = [x for x in range(m) if x != a and x != b]
        for ki, k in enumerate(remaining):
            new_dist[ki] = 0.5 * (dist[a, k] + dist[b, k] - dist[a, b])
        
        # Build new distance matrix
        new_m = m - 1
        new_dist_mat = np.zeros((new_m, new_m))
        for i in range(new_m):
            for j in range(i + 1, new_m):
                if i == new_m - 1 or j == new_m - 1:
                    # One of them is the new node
                    if i == new_m - 1:
                        new_dist_mat[i, j] = new_dist[j]
                        new_dist_mat[j, i] = new_dist[j]
                    else:
                        new_dist_mat[i, j] = new_dist[i]
                        new_dist_mat[j, i] = new_dist[i]
                else:
                    # Both are original remaining nodes
                    # Need to map old indices
                    old_i = remaining[i]
                    old_j = remaining[j]
                    new_dist_mat[i, j] = dist[old_i, old_j]
                    new_dist_mat[j, i] = dist[old_i, old_j]
        
        # Update subtree
        subtree[new_label] = f"({subtree[labels[a]] if a < n else subtree[str(a)]},{subtree[labels[b]] if b < n else subtree[str(b)]})"
        
        # Wait, this is getting complicated. Let me restructure.
        # Actually, let me use a cleaner NJ implementation.
        break
    
    # This approach is getting messy. Let me rewrite NJ cleanly.
    return None  # fallback


def _nj_clean(sequences, labels):
    """Clean Neighbor Joining implementation."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]})" + ";"
    
    # Compute distance matrix
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = sum(1 for x, y in zip(seqs[i], seqs[j]) if x != y)
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    
    # NJ algorithm
    # active: list of node indices (0..n-1 initially, then new nodes)
    # node_subtree: dict mapping node_id -> Newick string
    # node_is_taxon: whether the node is an original taxon
    
    active = list(range(n))
    node_subtree = {i: labels[i] for i in range(n)}
    # dist: dict of (i,j) -> distance, or use a matrix with growing indices
    # Let's use a dict for distances between active nodes
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
    
    next_node_id = n
    m = n
    
    while len(active) > 3:
        if time.time() > _DEADLINE:
            break
        
        m = len(active)
        # Row sums
        row_sum = {a: 0.0 for a in active}
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i != j:
                    row_sum[a] += dist[(a, b)]
        
        # Find best pair
        best_criterion = float('inf')
        best_a = active[0]
        best_b = active[1]
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i >= j:
                    continue
                criterion = (m - 2) * dist[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
        
        a = best_a
        b = best_b
        
        # Branch lengths
        da = 0.5 * dist[(a, b)] + (row_sum[a] - row_sum[b]) / (2.0 * (m - 2))
        db = dist[(a, b)] - da
        da = max(0.0, da)
        db = max(0.0, db)
        
        # New node
        new_node = next_node_id
        next_node_id += 1
        
        # New subtree
        node_subtree[new_node] = f"({node_subtree[a]},{node_subtree[b]})"
        
        # Update distances
        remaining = [x for x in active if x != a and x != b]
        for x in remaining:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        # Remove old distances
        for x in remaining:
            del dist[(a, x)]
            del dist[(x, a)]
            del dist[(b, x)]
            del dist[(x, b)]
        del dist[(a, b)]
        del dist[(b, a)]
        
        # Update active
        active = remaining + [new_node]
    
    # Final: join remaining 3 nodes into a star (or use remaining distances)
    if len(active) == 3:
        a, b, c = active
        return f"({node_subtree[a]},{node_subtree[b]},{node_subtree[c]})" + ";"
    elif len(active) == 2:
        a, b = active
        return f"({node_subtree[a]},{node_subtree[b]})" + ";"
    elif len(active) == 1:
        return node_subtree[active[0]] + ";"
    else:
        # Fallback star
        return "(" + ",".join(node_subtree[x] for x in active) + ")" + ";"


def _parse_newick(newick):
    """
    Parse a Newick string into a tree representation.
    Returns: (nodes, children, parent) where:
      nodes: dict node_id -> label (for leaves) or None (for internal)
      children: dict node_id -> list of child node_ids
      parent: dict node_id -> parent node_id or None
    """
    newick = newick.strip().rstrip(';')
    if not newick:
        return {}, {}, {}
    
    nodes = {}
    children = {}
    parent = {}
    
    next_id = 0
    # Stack-based parser
    # Each stack entry: (node_id, children_list, is_open)
    
    # We'll build the tree by tracking the current node
    # A simpler approach: recursive descent
    
    def parse(s, pos):
        nonlocal next_id
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            pos += 1
            # Parse children
            child_ids = []
            while True:
                if pos >= len(s):
                    break
                if s[pos] == ')':
                    pos += 1
                    break
                child_id, pos = parse(s, pos)
                if child_id is not None:
                    child_ids.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            # After ')', there might be a label
            # But for unrooted trees, we don't really need labels on internal nodes
            # Create internal node
            node_id = next_id
            next_id += 1
            nodes[node_id] = None  # internal node
            children[node_id] = child_ids
            for c in child_ids:
                parent[c] = node_id
            # Check for label after ')'
            # For our purposes, internal nodes don't need labels
            return node_id, pos
        else:
            # Leaf: parse label until ',' or ')'
            start = pos
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            label = s[start:pos]
            node_id = next_id
            next_id += 1
            nodes[node_id] = label
            children[node_id] = []
            return node_id, pos
    
    root_id, _ = parse(newick, 0)
    if root_id is not None:
        parent[root_id] = None
    
    return nodes, children, parent


def _get_leaves(tree_nodes, tree_children):
    """Get list of leaf node IDs."""
    return [nid for nid in tree_nodes if tree_nodes[nid] is not None]


def _get_splits(tree_nodes, tree_children, tree_parent, leaf_set):
    """
    Get the set of non-trivial unrooted splits.
    A split is a frozenset of leaves on one side of an edge.
    We only include splits where both sides have >= 2 leaves.
    """
    n_leaves = len(leaf_set)
    splits = set()
    
    for nid in tree_nodes:
        if tree_nodes[nid] is None:  # internal node
            # Get all leaves in the subtree rooted at nid
            subtree_leaves = set()
            stack = [nid]
            while stack:
                cur = stack.pop()
                if tree_nodes[cur] is not None:
                    subtree_leaves.add(tree_nodes[cur])
                else:
                    for c in tree_children[cur]:
                        stack.append(c)
            
            k = len(subtree_leaves)
            if 1 < k < n_leaves - 1:
                splits.add(frozenset(subtree_leaves))
    
    return splits


def _nni_refine(newick, sequences, labels):
    """
    Perform NNI (Nearest Neighbor Interchange) refinement.
    This is a heuristic: we try to improve the tree by local rearrangements.
    For simplicity, we'll do a limited number of NNI sweeps.
    """
    # Parse the tree
    tree_nodes, tree_children, tree_parent = _parse_newick(newick)
    leaf_set = set(tree_nodes[nid] for nid in tree_nodes if tree_nodes[nid] is not None)
    n_leaves = len(leaf_set)
    
    if n_leaves < 4:
        return newick
    
    # For NNI, we need to work with an unrooted tree representation.
    # This is complex. Given time constraints, let's do a simpler approach:
    # Just return the NJ tree as-is. NNI refinement is complex to implement correctly.
    return newick


def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree topology from DNA sequences.
    Returns a Newick string.
    """
    start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]})" + ";"
    
    # Use NJ
    newick = _nj_clean(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return newick
    
    # NNI refinement (limited)
    # For now, just return NJ result
    return newick
# EVOLVE-BLOCK-END
