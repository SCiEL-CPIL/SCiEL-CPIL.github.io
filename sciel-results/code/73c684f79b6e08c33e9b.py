    # Verification
    # Check if U, V, W are valid
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Target T[i,j,k] = 1 for all i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Calculate dot product for this specific (i,j,k)
                # U[u_idx, :] is shape (R,)
                # V[v_idx, :] is shape (R,)
                # W[w_idx, :] is shape (R,)
                # Sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                prod = U[u_idx, :rank_counter] * V[v_idx, :rank_counter] * W[w_idx, :rank_counter]
                val = np.sum(prod)
                if abs(val - 1.0) > 1e-9:
                    max_err = max(max_err, abs(val - 1.0))
                    
    if max_err > 1e-9:
        # Fallback
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
