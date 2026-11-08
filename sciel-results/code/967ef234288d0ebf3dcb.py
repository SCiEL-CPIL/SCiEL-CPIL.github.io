def _try_reduce_rank_local_search(U, V, W, n, m, p, target_rank, deadline):
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W

    current_U = Ua.copy()
    current_V = Va.copy()
    current_W = Wa.copy()
    current_R = R0
    
    factors = ['U', 'V', 'W']
    
    while current_R > target_rank:
        if time.time() > deadline - 5:
            break
            
        dropped = False
        for t_drop in range(current_R):
            if time.time() > deadline - 5:
                break
                
            backup_U = current_U.copy()
            backup_V = current_V.copy()
            backup_W = current_W.copy()
            
            current_U[:, t_drop] = 0
            current_V[:, t_drop] = 0
            current_W[:, t_drop] = 0
            
            success = False
            
            for factor in factors:
                if time.time() > deadline - 5:
                    break
                    
                n_vars = 0
                # Determine variable size
                if factor == 'U':
                    n_vars = n * m * (current_R - 1)
                elif factor == 'V':
                    n_vars = m * p * (current_R - 1)
                else:
                    n_vars = p * n * (current_R - 1)
                    
                n_eqs = n * m * p
                A = np.zeros((n_eqs, n_vars))
                b = np.ones(n_eqs)
                
                # Fill A
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for t in range(current_R):
                                if t == t_drop:
                                    continue
                                if t < t_drop:
                                    t_new = t
                                else:
                                    t_new = t - 1
                                    
                                if factor == 'U':
                                    coeff = current_V[j * p + k, t] * current_W[k * n + i, t]
                                    var_idx = (i * m + j) * (current_R - 1) + t_new
                                elif factor == 'V':
                                    coeff = current_U[i * m + j, t] * current_W[k * n + i, t]
                                    var_idx = (j * p + k) * (current_R - 1) + t_new
                                else:
                                    coeff = current_U[i * m + j, t] * current_V[j * p + k, t]
                                    var_idx = (k * n + i) * (current_R - 1) + t_new
                                    
                                A[eq_idx, var_idx] += coeff
                
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if len(residuals) > 0 and residuals[0] > 1e-8:
                        continue
                    
                    x_rounded = np.round(2 * x) / 2.0
                    if np.allclose(x, x_rounded, atol=1e-6):
                        # Reconstruct factor
                        if factor == 'U':
                            for i in range(n):
                                for j in range(m):
                                    for t in range(current_R):
                                        if t == t_drop: continue
                                        t_new = t if t < t_drop else t - 1
                                        current_U[i * m + j, t] = x_rounded[(i * m + j) * (current_R - 1) + t_new]
                        elif factor == 'V':
                            for j in range(m):
                                for k in range(p):
                                    for t in range(current_R):
                                        if t == t_drop: continue
                                        t_new = t if t < t_drop else t - 1
                                        current_V[j * p + k, t] = x_rounded[(j * p + k) * (current_R - 1) + t_new]
                        else:
                            for k in range(p):
                                for i in range(n):
                                    for t in range(current_R):
                                        if t == t_drop: continue
                                        t_new = t if t < t_drop else t - 1
                                        current_W[k * n + i, t] = x_rounded[(k * n + i) * (current_R - 1) + t_new]
                        
                        if _validate(current_U.tolist(), current_V.tolist(), current_W.tolist(), n, m, p):
                            mask = np.ones(current_R, dtype=bool)
                            mask[t_drop] = False
                            current_U = current_U[:, mask]
                            current_V = current_V[:, mask]
                            current_W = current_W[:, mask]
                            current_R -= 1
                            dropped = True
                            success = True
                            break
                except Exception:
                    pass
                
            if success:
                break
            else:
                # Restore if this t_drop failed
                current_U = backup_U
                current_V = backup_V
                current_W = backup_W

        if not dropped:
            break

    return current_U.tolist(), current_V.tolist(), current_W.tolist()
