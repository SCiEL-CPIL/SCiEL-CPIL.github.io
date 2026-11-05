"""
Phylogenetic topology recovery using Multi-Start Neighbor Joining.
The tree with the minimum sum of branch lengths (Minimum Evolution Score) 
is selected from multiple NJ runs, some of which use subsampled sites 
to escape local optima.
"""
import math
import time
import numpy as np

# Deadline guard: 180s CPU budget for 20 instances, so ~9s per instance.
# We will use a safety margin of 8.5s per instance.
TIME_LIMIT_PER_INSTANCE = 8.5

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = 0
    for i in range(n):
        if a[i] != b[i]:
            diff += 1
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - (4.0/3.0) * p)

def _compute_distance_matrix(sequences, labels):
    """Compute the pairwise JC-corrected distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _nj_build(dist, labels):
    """
    Neighbor Joining algorithm.
    Returns a nested tuple structure representing the tree.
    Leaf: string label
    Internal Node: (left_child, right_child, len_left, len_right)
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1], 0.0, 0.0)
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in nodes}
    current_dist = dist.copy()
    
    while len(nodes) > 2:
        m = len(nodes)
        row_sums = np.array([sum(current_dist[nodes[i], nodes[j]] for j in range(m) if j != i) for i in range(m)])
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * current_dist[nodes[i], nodes[j]] - row_sums[i] - row_sums[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        
        d_ab = current_dist[a, b]
        r_a = row_sums[best_i]
        r_b = row_sums[best_j]
        
        len_a = 0.5 * d_ab + (r_a - r_b) / (2 * (m - 2))
        len_b = d_ab - len_a
        
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        new_node_idx = max(nodes) + 1
        trees[new_node_idx] = (trees[a], trees[b], len_a, len_b)
        
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            new_d = 0.5 * (current_dist[a, x] + current_dist[b, x] - d_ab)
            current_dist[new_node_idx, x] = new_d
            current_dist[x, new_node_idx] = new_d
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_node_idx)
        
    a, b = nodes
    d_ab = current_dist[a, b]
    len_a = 0.5 * d_ab
    len_b = 0.5 * d_ab
    trees[new_node_idx] = (trees[a], trees[b], len_a, len_b)
    
    return trees[new_node_idx]

def _me_score(tree):
    """Calculate the Minimum Evolution Score (sum of branch lengths)."""
    total_len = 0.0
    def traverse(node):
        nonlocal total_len
        if isinstance(node, str):
            return
        left, right, len_l, len_r = node
        total_len += len_l + len_r
        traverse(left)
        traverse(right)
        
    traverse(tree)
    return total_len

def _tree_to_newick(node):
    """Convert the nested tuple tree to a Newick string."""
    if isinstance(node, str):
        return node
    left, right, len_l, len_r = node
    return f"({_tree_to_newick(left)}:{len_l:.4f},{_tree_to_newick(right)}:{len_r:.4f})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    start_time = time.time()
    
    # 1. Compute Full Distance Matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # 2. Initial NJ
    tree = _nj_build(dist, labels)
    best_tree = tree
    best_score = _me_score(tree)
    
    # 3. Multi-Start with Subsampling
    seq_len = len(sequences[labels[0]])
    
    # Determine number of starts based on time and problem size
    # NJ is O(N^3). For N=32, it's very fast.
    # We can run several starts if we have time.
    
    num_starts = 5
    if n < 10:
        num_starts = 10
    elif n > 25:
        num_starts = 3

    for k in range(1, num_starts):
        if time.time() - start_time > TIME_LIMIT_PER_INSTANCE * 0.8:
            break
            
        # Subsample sites to create a noisy distance matrix
        # This helps escape local optima of the greedy NJ
        if seq_len > 100:
            subset_size = max(50, seq_len // 2)
            subset_indices = np.random.choice(seq_len, size=subset_size, replace=False)
            sub_seqs = {l: sequences[l][subset_indices] for l in labels}
        else:
            # If sequences are short, subsampling might be too aggressive.
            # Just use the full matrix but with a slight perturbation?
            # Or just run NJ again (it's deterministic, so same result).
            # So for short sequences, multi-start is less useful unless we perturb distances.
            # Let's add small noise to distances for diversity.
            sub_dist = dist + np.random.normal(0, 0.01, dist.shape)
            sub_dist = (sub_dist + sub_dist.T) / 2
            np.fill_diagonal(sub_dist, 0)
            sub_tree = _nj_build(sub_dist, labels)
            score = _me_score(sub_tree)
            if score < best_score:
                best_score = score
                best_tree = sub_tree
            continue

        sub_dist = _compute_distance_matrix(sub_seqs, labels)
        sub_tree = _nj_build(sub_dist, labels)
        
        # Score the sub_tree using the FULL distance matrix's ME score?
        # No, ME score is intrinsic to the tree (sum of its branch lengths).
        # The tree derived from subsampled data might have different branch lengths.
        # We should compare them using a metric that reflects topological accuracy.
        # Minimum Evolution is a proxy. A better proxy is the Quartet Score.
        # But for now, let's assume the tree with the lowest ME score is better.
        # This is a heuristic.
        
        score = _me_score(sub_tree)
        if score < best_score:
            best_score = score
            best_tree = sub_tree

    newick = _tree_to_newick(best_tree)
    return newick + ";"

np.random.seed(42)
