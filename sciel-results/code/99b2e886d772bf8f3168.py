def local_search_pass_vectorized(x, time_budget):
    R = _compute_autocorr(x)
    current_max = np.max(np.abs(R[1:]))
    
    # Precompute index matrix for x_{j-k} and x_{j+k}
    # idx[j, k] = (j - k) % N
    idx = np.arange(N)
    # M_minus[j, k] = x[(j-k)%N]
    # We can create a matrix of indices
    # j is row, k is col
    # j-k mod N
    # This is equivalent to np.roll(x, j) for each row j?
    # Actually, M_minus is a circulant matrix.
    # We can just compute it once? No, x changes.
    # But the indices are fixed.
    
    # Let's just compute the delta matrix directly.
    # For a fixed x, we can evaluate all flips.
    
    while (time.time() - start_time) < (start_time + time_budget):
        R = _compute_autocorr(x)
        current_max = np.max(np.abs(R[1:]))
        
        # Vectorized flip evaluation
        # x_j is x[:, None]
        # x_{j-k} is x[(j-k)%N]
        # x_{j+k} is x[(j+k)%N]
        
        # Create index arrays
        j_idx = np.arange(N)[:, None]
        k_idx = np.arange(N)[None, :]
        
        # Compute indices
        # To avoid modulo in large array, we can use np.roll logic or just modulo.
        # Modulo is fine.
        idx_minus = (j_idx - k_idx) % N
        idx_plus = (j_idx + k_idx) % N
        
        x_minus = x[idx_minus]
        x_plus = x[idx_plus]
        
        delta_R = -2.0 * x[:, None] * (x_minus + x_plus)
        
        new_R = R[None, :] + delta_R
        
        # Exclude k=0 (column 0)
        new_max = np.max(np.abs(new_R[:, 1:]), axis=1)
        
        best_j = np.argmin(new_max)
        best_val = new_max[best_j]
        
        if best_val < current_max - 1e-9:
            x[best_j] *= -1
            # Update current_max? It will be recomputed at start of loop.
            # But we can update it here to break early if we want?
            # No, just let the loop handle it.
        else:
            break
            
    return x, current_max
