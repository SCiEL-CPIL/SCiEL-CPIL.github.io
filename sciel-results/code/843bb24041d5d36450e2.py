import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial rank
    trivial_rank = n * m * p
    
    # We know a bound of 85 exists (49 + 16 + 20).
    # We will try to find a decomposition with rank < trivial_rank.
    # We start from a lower bound and go up.
    # Lower bound is max(n*m, m*p, p*n) = 25.
    # But 25 is likely not achievable.
    # We try a few ranks.
    
    ranks_to_try = [85, 90, 95]
    
    for R in ranks_to_try:
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Precompute indices for vectorization
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # We will use vectorized ALS
        # For each iteration:
        # 1. Update U: For each (i,j), U[i,j,:] = (V_j * W_i^T) / ||V_j * W_i^T||^2 ?
        #    Actually, we want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k}? No.
        #    The target is C_ik = sum_j A_ij B_jk.
        #    The tensor T_ijk = delta_{?}
        #    The equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if the term A_ij B_jk is present?
        #    Yes, the tensor is 1 for all (i,j,k) because every term A_ij B_jk contributes to C_ik.
        #    Wait, C_ik = sum_j A_ij B_jk.
        #    So the coefficient of A_ij B_jk in C_ik is 1.
        #    So T_ijk = 1 for all i, j, k.
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(100): # Limit iterations
            # Update U
            # A_u[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # Vectorized:
            # V is m*p x R. Reshape to m x p x R.
            # W is p*n x R. Reshape to p x n x R.
            # We need for each i, j: sum_k V[j, k, r] * W[k, i, r].
            # This is a contraction over k.
            # V_reshaped: (m, p, R)
            # W_reshaped: (p, n, R)
            # We want result (n, m, R).
            # A_u[i, j, r] = sum_k V[j, k, r] W[k, i, r].
            # This is like a matrix multiplication if we fix r?
            # V[j, :, r] is a vector of length p.
            # W[:, i, r] is a vector of length p.
            # Dot product.
            # We can do this for all r at once using einsum or broadcasting.
            # V_reshaped: (m, p, R)
            # W_reshaped: (p, n, R)
            # We want (n, m, R).
            # Swap axes of W to (n, p, R)?
            # W_reshaped.T: (R, p, n).
            # Let's use np.einsum('jkr,kin->ijnr', V_reshaped, W_reshaped) -> (i, j, n, R)? No.
            # We want sum over k.
            # V[j, k, r] * W[k, i, r].
            # Indices: j, k, r and k, i, r.
            # Result: i, j, r.
            # np.einsum('jkr,kin->ijr', V_reshaped, W_reshaped)
            
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            A_u = np.einsum('jkr,kin->ijr', V_r, W_r) # Shape (n, m, R)
            A_u_flat = A_u.reshape(n * m, R)
            
            norms_sq = np.sum(A_u_flat * A_u_flat, axis=1)
            norms_sq[norms_sq == 0] = 1e-9
            U_new = A_u_flat / norms_sq[:, np.newaxis]
            
            # Update V
            # A_v[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            # U_r: (n, m, R)
            # W_r: (p, n, R)
            # We want (m, p, R).
            # U[i, j, r] * W[k, i, r]. Sum over i.
            # np.einsum('ijr,kin->jkr', U_r, W_r)
            
            U_r = U_new.reshape(n, m, R)
            A_v = np.einsum('ijr,kin->jkr', U_r, W_r) # Shape (m, p, R)
            A_v_flat = A_v.reshape(m * p, R)
            
            norms_sq_v = np.sum(A_v_flat * A_v_flat, axis=1)
            norms_sq_v[norms_sq_v == 0] = 1e-9
            V_new = A_v_flat / norms_sq_v[:, np.newaxis]
            
            # Update W
            # A_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # U_r: (n, m, R)
            # V_r: (m, p, R)
            # We want (p, n, R).
            # U[i, j, r] * V[j, k, r]. Sum over j.
            # np.einsum('ijr,jkr->kir', U_r, V_r)
            
            V_r_new = V_new.reshape(m, p, R)
            A_w = np.einsum('ijr,jkr->kir', U_r, V_r_new) # Shape (p, n, R)
            A_w_flat = A_w.reshape(p * n, R)
            
            norms_sq_w = np.sum(A_w_flat * A_w_flat, axis=1)
            norms_sq_w[norms_sq_w == 0] = 1e-9
            W_new = A_w_flat / norms_sq_w[:, np.newaxis]
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate Error
            # err = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            # val[i,j,k] = sum_r U[i,j,r]V[j,k,r]W[k,i,r]
            # U_r: (n, m, R), V_r: (m, p, R), W_r: (p, n, R)
            # np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            val = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            err = np.sum((val - 1.0) ** 2)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-6:
                break

        if best_error < 1e-4:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            is_exact = True
            U_r = U_int.reshape(n, m, R)
            V_r = V_int.reshape(m, p, R)
            W_r = W_int.reshape(p, n, R)
            val = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if np.max(np.abs(val - 1.0)) > 1e-9:
                is_exact = False
            
            if is_exact:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
