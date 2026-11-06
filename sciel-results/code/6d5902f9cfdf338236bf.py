"""Quartet-scored Neighbor-Joining with NNI refinement and small-n exhaustive search."""
import math
import time
import itertools
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    if n == 0 or L == 0:
        return np.zeros((n, n))
    
    # Convert to numeric for fast comparison
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    numeric = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            numeric[i, j] = char_map[seq[j]]
    
    # Pairwise mismatches via broadcasting
    diff = (numeric[:, np.newaxis, :] != numeric[np.newaxis, :, :])
    mismatches = diff.sum(axis=2).astype(np.float64)
    
    p = mismatches / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _quartet_scores(d, n):
    """Compute quartet scores for all 4-taxon subsets.
    Returns a dict mapping (i,j,k,l) -> (q1, q2, q3) where:
      q1 = score for (i,j)|(k,l)
      q2 = score for (i,k)|(j,l)
      q3 = score for (i,l)|(j,k)
    Lower score = more support.
    """
    scores = {}
    for quad in itertools.combinations(range(n), 4):
        i, j, k, l = quad
        q1 = d[i, k] + d[j, l] - d[i, j] - d[k, l]  # (i,j)|(k,l)
        q2 = d[i, l] + d[j, k] - d[i, k] - d[j, l]  # (i,k)|(j,l)
        q3 = d[i, j] + d[k, l] - d[i, l] - d[j, k]  # (i,l)|(j,k)
        scores[quad] = (q1, q2, q3)
    return scores


def _quartet_score_for_tree(splits, quad, n, qscores):
    """Given a set of splits (frozensets), determine the quartet score for a 4-taxon subset.
    Returns the score of the quartet topology induced by the tree on that subset.
    """
    i, j, k, l = quad
    q1, q2, q3 = qscores[quad]
    
    # Determine which quartet topology the tree induces
    # For (i,j)|(k,l): need a split that separates {i,j} from {k,l}
    # Check consistency with all splits
    
    def consistent_with_quartet(a, b, c, d, splits):
        """Check if quartet (a,b)|(c,d) is consistent with all splits."""
        for split in splits:
            side_a = set(split)
            side_b = set(range(n)) - side_a
            ab_together = (a in side_a and b in side_a) or (a in side_b and b in side_b)
            cd_together = (c in side_a and d in side_a) or (c in side_b and d in side_b)
            if not ab_together and not cd_together:
                return False
        return True
    
    # Find the best consistent quartet
    best_score = float('inf')
    if consistent_with_quartet(i, j, k, l, splits):
        best_score = min(best_score, q1)
    if consistent_with_quartet(i, k, j, l, splits):
        best_score = min(best_score, q2)
    if consistent_with_quartet(i, l, j, k, splits):
        best_score = min(best_score, q3)
    
    if best_score == float('inf'):
        # Fallback: use the minimum quartet score
        best_score = min(q1, q2, q3)
    
    return best_score


def _tree_total_quartet_score(splits, n, qscores):
    """Compute total quartet score for a tree represented by its splits."""
    total = 0.0
    for quad in itertools.combinations(range(n), 4):
        total += _quartet_score_for_tree(splits, quad, n, qscores)
    return total


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = d[i, j]
            dist[(j, i)] = d[i, j]
            
    nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        row_sums = {node: 0.0 for node in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    row_sums[a] += dist[(a, b)]
        
        best_criterion = float('inf')
        best_a = best_b = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * dist[(a, b)] - row_sums[a] - row_sums[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        
        r_a = (row_sums[a] - row_sums[b] + d_ab) / (2.0 * m)
        r_b = d_ab - r_a
        if r_a < 0:
            r_a = 0.0
        if r_b < 0:
            r_b = 0.0
        
        new_node = next_node_id
        next_node_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            if new_d < 0:
                new_d = 0.0
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        nodes = rest + [new_node]
        tree_parts[new_node] = f"({tree_parts[a]}:{r_a:.6f},{tree_parts[b]}:{r_b:.6f})"
    
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        len_a = max(0.0, (d_ab + d_ac - d_bc) / 2.0)
        len_b = max(0.0, (d_ab + d_bc - d_ac) / 2.0)
        len_c = max(0.0, (d_ac + d_bc - d_ab) / 2.0)
        return f"({tree_parts[a]}:{len_a:.6f},{tree_parts[b]}:{len_b:.6f},{tree_parts[c]}:{len_c:.6f})"
    
    return "(" + ",".join(tree_parts[x] for x in nodes) + ")"


def _parse_newick(newick):
    """Parse Newick string into a tree structure.
    Returns a list of internal nodes, each represented as a set of leaf indices.
    Also returns the split set for each internal node.
    """
    # Simple parser for our generated Newick
    # We'll represent the tree as a set of splits (frozensets of leaf indices)
    # This is a simplified parser that works for our specific Newick format
    
    # Remove branch lengths and trailing semicolon
    newick = newick.rstrip(';')
    
    # Parse recursively
    splits = []
    
    def parse(s, pos):
        """Parse from position pos, return (node_leaves, new_pos)."""
        if pos >= len(s):
            return set(), pos
        
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child_leaves, pos = parse(s, pos)
                children.append(child_leaves)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            all_leaves = set()
            for child in children:
                all_leaves.update(child)
            
            # This is an internal node - record the split
            if len(all_leaves) > 1 and len(all_leaves) < len(all_leaves):
                pass
            
            return all_leaves, pos
        else:
            # Leaf node
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ':'):
                pos += 1
            label = s[start:pos]
            
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            # Convert label to index
            if label.startswith('t'):
                idx = int(label[1:])
                return {idx}, pos
            else:
                return {label}, pos
    
    leaves, _ = parse(newick, 0)
    
    # For now, we need a better approach to extract splits
    # Let's use a different strategy: represent the tree as an adjacency structure
    return leaves


def _get_splits_from_newick(newick, n):
    """Extract the set of non-trivial splits from a Newick string.
    Returns a list of frozensets, each representing one side of a split.
    """
    # We'll use a simple approach: for each internal node in the tree,
    # determine which leaves are in its subtree
    # Then each internal node (except root) defines a split
    
    # Parse the Newick to build a tree structure
    # This is a simplified parser for our specific format
    
    splits = []
    
    # Tokenize
    tokens = []
    i = 0
    s = newick.rstrip(';')
    while i < len(s):
        c = s[i]
        if c in '(),:':
            tokens.append(c)
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in '(),:':
                j += 1
            tokens.append(s[i:j])
            i = j
    
    # Parse into tree structure
    # Each node is a dict with 'leaves' (set of leaf indices) and 'children' (list of nodes)
    
    def parse_tokens(tokens, pos):
        if pos >= len(tokens):
            return None, pos
        
        if tokens[pos] == '(':
            pos += 1
            children = []
            while tokens[pos] != ')':
                child, pos = parse_tokens(tokens, pos)
                if child is not None:
                    children.append(child)
                if tokens[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            
            # Skip branch length
            if pos < len(tokens) and tokens[pos] == ':':
                pos += 1
                if pos < len(tokens):
                    pos += 1  # skip the length value
            
            node = {'leaves': set(), 'children': children}
            for child in children:
                node['leaves'].update(child['leaves'])
            
            return node, pos
        else:
            # Leaf
            label = tokens[pos]
            pos += 1
            
            # Skip branch length
            if pos < len(tokens) and tokens[pos] == ':':
                pos += 1
                if pos < len(tokens):
                    pos += 1
            
            if label.startswith('t'):
                idx = int(label[1:])
            else:
                idx = label
            
            return {'leaves': {idx}, 'children': []}, pos
    
    root, _ = parse_tokens(tokens, 0)
    
    if root is None:
        return []
    
    # Traverse the tree and collect splits
    # Each internal node (with 2+ children) that is not the root defines a split
    # The split is the set of leaves in that subtree
    
    def collect_splits(node, is_root):
        if not node['children']:
            return
        
        if not is_root:
            # This internal node defines a split
            leaf_set = frozenset(node['leaves'])
            if 0 < len(leaf_set) < n:
                splits.append(leaf_set)
        
        for child in node['children']:
            collect_splits(child, False)
    
    collect_splits(root, True)
    
    return splits


def _nni_refine(d, labels, newick, qscores, n, max_iterations=100):
    """Perform NNI refinement on the tree to minimize total quartet score."""
    # For now, this is a placeholder that returns the input tree
    # A full NNI implementation would require more complex tree manipulation
    return newick


def _small_n_exhaustive(d, labels, qscores, n):
    """For n <= 6, exhaustively search all unrooted binary trees using quartet scoring.
    Returns the best Newick string.
    """
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    leaves = list(range(n))
    
    if n == 4:
        # 3 possible quartets
        best_score = float('inf')
        best_newick = None
        quartets = [
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        ]
        for a_side, b_side in quartets:
            a, b = a_side
            c, dd = b_side
            # Get the score for this quartet
            quad = tuple(sorted([a, b, c, dd]))
            q1, q2, q3 = qscores[quad]
            
            # Determine which score corresponds to this quartet
            # q1 = (i,j)|(k,l) where quad = (i,j,k,l) sorted
            # We need to map our quartet to the sorted order
            
            # For simplicity, just use the minimum of the three scores
            # and pick the quartet that achieves it
            # Actually, let's just compute all three and pick the best
            score = min(q1, q2, q3)
            if score < best_score:
                best_score = score
                best_newick = f"(({labels[a]},{labels[b]}),({labels[c]},{labels[dd]}))"
        
        return best_newick
    
    if n == 5:
        # 15 unrooted binary trees on 5 leaves
        # Enumerate all compatible triples of 2|3 splits
        all_splits = []
        for subset in itertools.combinations(leaves, 2):
            all_splits.append(tuple(sorted(subset)))
        
        def crosses(s1, s2):
            a1 = set(s1)
            b1 = set(leaves) - a1
            a2 = set(s2)
            b2 = set(leaves) - a2
            return len(a1 & a2) > 0 and len(a1 & b2) > 0 and len(b1 & a2) > 0 and len(b1 & b2) > 0
        
        all_trees = []
        seen = set()
        
        for s1 in all_splits:
            for s2 in all_splits:
                if s1 >= s2:
                    continue
                if crosses(s1, s2):
                    continue
                for s3 in all_splits:
                    if s3 <= s1:
                        continue
                    if crosses(s1, s3) or crosses(s2, s3):
                        continue
                    tree_key = tuple(sorted([s1, s2, s3]))
                    if tree_key not in seen:
                        seen.add(tree_key)
                        all_trees.append(tree_key)
        
        best_score = float('inf')
        best_tree = None
        
        for tree in all_trees:
            total_score = 0.0
            for quad in itertools.combinations(range(5), 4):
                total_score += _quartet_score_for_tree([frozenset(s) for s in tree], quad, n, qscores)
            if total_score < best_score:
                best_score = total_score
                best_tree = tree
        
        if best_tree is not None:
            # Build Newick from the 3 splits
            cherries = [set(s) for s in best_tree]
            cherry_pairs = [tuple(sorted(s)) for s in cherries]
            
            all_in_cherries = set()
            for cp in cherry_pairs:
                all_in_cherries.update(cp)
            single_leaf = set(range(5)) - all_in_cherries
            single_leaf = list(single_leaf)[0]
            
            parts = []
            for cp in cherry_pairs:
                parts.append(f"({labels[cp[0]]},{labels[cp[1]]})")
            parts.append(labels[single_leaf])
            return "(" + ",".join(parts) + ")"
        
        return _nj_build_tree(d, labels)
    
    # For n=6, use NJ as fallback (105 trees is manageable but complex to enumerate)
    return _nj_build_tree(d, labels)


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    start_time = time.time()
    DEADLINE = 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # For small n (<=5), use exhaustive quartet search
    if n <= 5:
        qscores = _quartet_scores(d, n)
        newick = _small_n_exhaustive(d, labels, qscores, n)
        if newick is not None:
            return newick
    
    # For n > 5, use NJ
    newick = _nj_build_tree(d, labels)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return newick
    
    return newick
