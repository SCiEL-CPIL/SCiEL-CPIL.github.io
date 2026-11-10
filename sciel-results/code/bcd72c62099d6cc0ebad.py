import numpy as np
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 240.0
    
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
    p = hamming.astype(np.float64) / L
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
    
    # BIONJ: variance-corrected neighbor joining
    # Compute variance estimates for each distance
    # For Jukes-Cantor model, Var(d_ij) ≈ (1/3) * (1/L) * (1/(1-p_ij)^2) * (1 + p_ij/(1-p_ij))
    # Simplified: use p_ij to estimate variance
    var_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            p_ij = p[i, j]
            if p_ij > 0 and p_ij < 0.75:
                # Approximate variance of JC-corrected distance
                var_ij = (1.0 / (3.0 * L)) * (1.0 / (1.0 - p_ij)**2) * (1.0 + p_ij / (1.0 - p_ij))
                var_matrix[i, j] = var_ij
                var_matrix[j, i] = var_ij
            else:
                var_matrix[i, j] = 1.0
                var_matrix[j, i] = 1.0
    
    # Neighbor Joining with BIONJ correction
    active = list(range(n_taxa))
    D = d_matrix.copy()
    V = var_matrix.copy()
    trees = {i: labels[i] for i in range(n_taxa)}
    next_id = n_taxa
    
    while len(active) > 3:
        n = len(active)
        r = D.sum(axis=1)
        
        # Standard NJ Q matrix
        Q = (n - 2) * D - r[:, None] - r[None, :]
        
        # BIONJ correction: weight by inverse variance
        # For each pair (i,j), the corrected Q is Q_ij / sqrt(Var(Q_ij))
        # Var(Q_ij) can be approximated from the variance of distances
        # Simplified BIONJ: use Q_ij * (1/sqrt(var_i + var_j)) where var_i = sum of variances to others
        
        # Compute row variance sums
        row_var = V.sum(axis=1)
        
        # BIONJ-corrected Q matrix
        # Q_bionj[i,j] = Q[i,j] / sqrt(row_var[i] + row_var[j])
        Q_bionj = Q.copy()
        for i in range(n):
            for j in range(i + 1, n):
                denom = np.sqrt(row_var[i] + row_var[j])
                if denom > 0:
                    Q_bionj[i, j] /= denom
                    Q_bionj[j, i] /= denom
        
        Q_masked = Q_bionj.copy()
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
        new_V = np.zeros((new_n, new_n), dtype=np.float64)
        
        for a_idx, a in enumerate(remaining_indices):
            for b_idx, b in enumerate(remaining_indices):
                new_D[a_idx, b_idx] = D[a, b]
                new_V[a_idx, b_idx] = V[a, b]
        
        # Update distances and variances for new node
        for a_idx, a in enumerate(remaining_indices):
            dist_new_a = 0.5 * (D[i, a] + D[j, a] - D[i, j])
            new_D[a_idx, new_n - 1] = dist_new_a
            new_D[new_n - 1, a_idx] = dist_new_a
            
            # Variance of new distance: Var(0.5*(d_ia + d_ja - d_ij))
            # Assuming independence: Var = 0.25 * (Var(ia) + Var(ja) + Var(ij))
            var_new_a = 0.25 * (V[i, a] + V[j, a] + V[i, j])
            new_V[a_idx, new_n - 1] = var_new_a
            new_V[new_n - 1, a_idx] = var_new_a
        
        new_D[new_n - 1, new_n - 1] = 0.0
        new_V[new_n - 1, new_n - 1] = 0.0
        
        active = [active[k] for k in remaining_indices] + [new_id]
        D = new_D
        V = new_V
    
    # Final 3 nodes
    t0 = trees[active[0]]
    t1 = trees[active[1]]
    t2 = trees[active[2]]
    final_tree = f"(({t0},{t1}),{t2})"
    
    return final_tree + ";"
