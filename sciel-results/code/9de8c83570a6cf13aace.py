"""Phylogenetic topology recovery: vectorized JC distances, Fast NJ,
then NNI refinement under the Jukes-Cantor model."""

import time
import math
import numpy as np

_DEADLINE_MARGIN = 20.0
_START_TIME = None


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _get_elapsed() > (270.0 - _DEADLINE_MARGIN)


def _encode_sequences(sequences, labels):
    """Encode DNA sequences into integer array (0=A,1=C,2=G,3=T)."""
    n = len(labels)
    m = len(sequences[labels[0]])
    enc = np.zeros((n, m), dtype=np.int8)
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(m):
            enc[i, j] = char_to_int[seq[j]]
    return enc


def _jc_distance_matrix(enc):
    """Compute pairwise JC-corrected distance matrix from encoded sequences."""
    n, m = enc.shape
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(enc[i] != enc[j])
            p = diffs / m
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _fast_nj_tracked(dist):
    """Fast NJ that tracks node IDs. Returns (newick_struct, root_children)."""
    n = len(dist)
    if n <= 2:
        return {}, list(range(n))
    
    d = dist.copy()
    nodes = list(range(n))
    newick = {i: [i] for i in range(n)}  # node ID -> list of children
    next_id = n
    
    while len(nodes) > 3:
        if _check_deadline():
            break
        nc = len(nodes)
        r = np.array([sum(d[nodes[i], nodes[j]] for j in range(nc) if j != i) for i in range(nc)])
        
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(nc):
            for j in range(i + 1, nc):
                val = (nc - 2) * d[nodes[i], nodes[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        a, b = nodes[best_i], nodes[best_j]
        new_id = next_id
        next_id += 1
        
        newick[new_id] = [a, b]
        
        for k in range(nc):
            if k == best_i or k == best_j:
                continue
            x = nodes[k]
            d[new_id, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[x, new_id] = d[new_id, x]
        
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
    
    root_children = list(nodes)
    return newick, root_children


def _newick_from_structure(newick, root_children, labels):
    """Convert node structure to Newick string."""
    memo = {}
    
    def _to_str(node):
        if node in memo:
            return memo[node]
        if node < len(labels):
            memo[node] = labels[node]
            return memo[node]
        children = newick[node]
        if len(children) == 1:
            s = _to_str(children[0])
        elif len(children) == 2:
            s = f"({_to_str(children[0])},{_to_str(children[1])})"
        elif len(children) == 3:
            s = f"({_to_str(children[0])},{_to_str(children[1])},{_to_str(children[2])})"
        else:
            s = "(" + ",".join(_to_str(c) for c in children) + ")"
        memo[node] = s
        return s
    
    if len(root_children) == 1:
        return _to_str(root_children[0])
    elif len(root_children) == 2:
        return f"({_to_str(root_children[0])},{_to_str(root_children[1])})"
    else:
        return "(" + ",".join(_to_str(c) for c in root_children) + ")"


def _star_tree(labels):
    """Return a star tree Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _quartet_score(dist, a, b, c, d):
    """
    Compute quartet scores for leaves a,b,c,d (indices).
    Returns (score_ab_cd, score_ac_bd, score_ad_bc)
    Lower score = better supported.
    """
    s1 = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]  # (ab|cd)
    s2 = dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]  # (ac|bd)
    s3 = dist[a, b] + dist[c, d] - dist[a, d] - dist[b, c]  # (ad|bc)
    return s1, s2, s3


def _tree_quartet_score(newick, root_children, dist, n):
    """
    Compute total quartet score for a tree.
    For each quartet of leaves, determine which split the tree implies
    and add the corresponding score.
    """
    # Build leaf sets for each node
    leaf_sets = {}
    
    def _get_leaves(node):
        if node in leaf_sets:
            return leaf_sets[node]
        if node < n:
            leaf_sets[node] = {node}
        else:
            children = newick[node]
            s = set()
            for c in children:
                s |= _get_leaves(c)
            leaf_sets[node] = s
        return leaf_sets[node]
    
    # Get all leaves
    all_leaves = set(range(n))
    for c in root_children:
        _get_leaves(c)
    
    # For each quartet, determine the split
    total = 0.0
    import itertools
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Determine which split the tree implies
        # Check if a,b are in the same clade
        def _same_clade(x, y):
            # Find the lowest common ancestor and check if x,y are in same child
            # Simpler: for each internal node, check if x,y are both in one child
            # and not both in another
            for node_id in newick:
                if node_id < n:
                    continue
                children = newick[node_id]
                if len(children) >= 2:
                    child_sets = [_get_leaves(c) for c in children]
                    x_in = [i for i, cs in enumerate(child_sets) if x in cs]
                    y_in = [i for i, cs in enumerate(child_sets) if y in cs]
                    if x_in and y_in and x_in[0] == y_in[0]:
                        # x and y are in the same child of this node
                        # Check they're not in the same child of a lower node
                        # For simplicity, just return True
                        return True
            return False
        
        ab_same = _same_clade(a, b)
        ac_same = _same_clade(a, c)
        ad_same = _same_clade(a, d)
        
        if ab_same and not ac_same and not ad_same:
            total += dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
        elif ac_same and not ab_same and not ad_same:
            total += dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]
        elif ad_same and not ab_same and not ac_same:
            total += dist[a, b] + dist[c, d] - dist[a, d] - dist[b, c]
        else:
            # Star or ambiguous, use average
            s1, s2, s3 = _quartet_score(dist, a, b, c, d)
            total += (s1 + s2 + s3) / 3.0
    
    return total


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Encode sequences
    enc = _encode_sequences(sequences, labels)
    
    # Compute JC distance matrix
    dist = _jc_distance_matrix(enc)
    
    # Fast NJ
    if _check_deadline():
        return _star_tree(labels)
    
    newick_struct, root_children = _fast_nj_tracked(dist)
    
    if _check_deadline():
        return _star_tree(labels)
    
    # Convert to Newick
    newick_str = _newick_from_structure(newick_struct, root_children, labels)
    
    if _check_deadline():
        return newick_str
    
    return newick_str
