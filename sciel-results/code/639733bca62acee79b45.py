# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Search for rank 59
    # Strategy: Remove one term, fix U and V, solve for W.
    
    U = np.array(best['U'], dtype=float)
    V = np.array(best['V'], dtype=float)
    W = np.array(best['W'], dtype=float)
    
    # Target tensor T
    # T[i, j, k] = 1 if ...
    # Actually, the tensor is defined by the schoolbook structure.
    # We can reconstruct T from the schoolbook.
    T = np.zeros((n, m, p))
    for r in range(base_rank):
        # Find non-zero indices in U[:, r], V[:, r], W[:, r]
        # In schoolbook, they are single 1s.
        # But we can just use the definition:
        # T[i, j, k] = 1 for all i, j, k? No.
        # The tensor is the sum of rank-1 terms.
        # We can compute T by summing the outer products.
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        pass
    
    # Compute T from schoolbook
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Try removing each term
    for r_remove in range(base_rank):
        if time.time() > deadline - 10:
            break
            
        U_mod = U.copy()
        V_mod = V.copy()
        W_mod = W.copy()
        
        # Zero out term r_remove
        U_mod[:, r_remove] = 0
        V_mod[:, r_remove] = 0
        W_mod[:, r_remove] = 0
        
        # Solve for W_mod
        # For each k, solve M_k w_k = T[:,:,k]
        # M_k is (n*m) x R
        # M_k[i*m+j, r] = U_mod[i, r] * V_mod[j, r]
        
        valid = True
        for k in range(p):
            M_k = np.zeros((n * m, base_rank))
            for i in range(n):
                for j in range(m):
                    M_k[i * m + j, :] = U_mod[i, :] * V_mod[j, :]
            
            # Solve M_k w_k = T[:,:,k].flatten()
            # Use least squares
            try:
                w_k, residuals, rank, sv = np.linalg.lstsq(M_k, T[:, :, k].flatten(), rcond=None)
                W_mod[k, :] = w_k
            except:
                valid = False
                break
        
        if not valid:
            continue
            
        # Verify exactness
        # Compute T_mod = einsum
        T_mod = np.einsum('ir,jr,kr->ijk', U_mod, V_mod, W_mod)
        
        if np.allclose(T_mod, T, atol=1e-9):
            # Check if coefficients are "nice" (integers/half-integers)
            # If not, we might still return it if the evaluator accepts floats?
            # The prompt says "integer/half-integer entries".
            # We should check if they are close to such values.
            
            # Check W_mod
            w_rounded = np.round(W_mod * 2) / 2
            if np.allclose(W_mod, w_rounded, atol=1e-6):
                # Found a valid half-integer solution
                best = {
                    "U": U_mod.tolist(),
                    "V": V_mod.tolist(),
                    "W": w_rounded.tolist()
                }
                best_rank = base_rank - 1
                return best
            else:
                # Not nice, but maybe we can polish?
                # For now, skip
                pass
                
    return best
# EVOLVE-BLOCK-END
