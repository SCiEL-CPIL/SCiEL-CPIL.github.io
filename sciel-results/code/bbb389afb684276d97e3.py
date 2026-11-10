import numpy as np
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 240.0  # 240s total, but we want margin for 20 instances
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return f"({labels[0]});" if n_taxa == 1 else ""
    
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Map sequences to integer arrays
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    seq_array = np.zeros((n_taxa, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j, char in enumerate(s):
            seq_array[i, j] = seq_map[char]
    
    # Compute pairwise Hamming distances
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)
    
    # Apply Jukes-Cantor correction
    p = hamming / L
    d_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    
    valid_mask = (p > 0) & (p < 0.75)
    d_valid = -0.75 * np.log(1 - (4.0 / 3.0) * p[valid_mask])
    d_matrix[valid_mask] = d_valid
    
    # For p >= 0.75, set to large value
    if np.any(valid_mask):
        large_val = np.max(d_valid) * 2.0
    else:
        large_val = 10.0
    d_matrix[p >= 0.75] = large_val
    np.fill_diagonal(d_matrix, 0.0)
    
    # Neighbor Joining
    active = list(range(n_taxa))
    D = d_matrix.copy()
    trees = {i: labels[i] for i in range(n_taxa)}
    next_id = n_taxa
    
    while len(active) > 3:
        n = len(active)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        
        Q_masked = Q.copy()
        np.fill_diagonal(Q_masked, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i, j = min_idx
        
        node_i = active[i]
        node_j = active[j]
        new_id = next_id
        next_id += 1
        
        trees[new_id] = f"({trees[node_i]},{trees[node_j]})"
        
        remaining_indices = [k for k in range(n) if k not in (i, j)]
        new_n = n - 1
        new_D = np.zeros((new_n, new_n), dtype=np.float64)
        
        for a_idx, a in enumerate(remaining_indices):
            for b_idx, b in enumerate(remaining_indices):
                new_D[a_idx, b_idx] = D[a, b]
        
        for a_idx, a in enumerate(remaining_indices):
            dist_new_a = 0.5 * (D[i, a] + D[j, a] - D[i, j])
            new_D[a_idx, new_n - 1] = dist_new_a
            new_D[new_n - 1, a_idx] = dist_new_a
        
        new_D[new_n - 1, new_n - 1] = 0.0
        active = [active[k] for k in remaining_indices] + [new_id]
        D = new_D
    
    # Final 3 nodes
    t0 = trees[active[0]]
    t1 = trees[active[1]]
    t2 = trees[active[2]]
    final_tree = f"(({t0},{t1}),{t2})"
    
    return final_tree + ";"
