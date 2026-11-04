import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    I, J, K = n*m, m*p, p*n
    
    # Build the exact multiplication tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    T = np.zeros((I, J, K), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # We want to find U, V, W such that T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # U shape (I, R), V shape (J, R), W shape (K, R)
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Try ranks from high to low? Or low to high?
    # Low to high allows finding the minimal rank if we succeed.
    # But ALS might get stuck in local minima for low ranks.
    # High ranks are easier to fit.
    # Let's try a range of ranks.
    # Lower bound is 18.
    # Schoolbook is 54.
    # Known upper bound is 46.
    # Let's try 45 down to 19.
    
    ranks_to_try = list(range(45, 18, -1))
    
    for R in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Initialize with random values
        # Use a few restarts for each rank
        found = False
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R + restart)
            U = np.random.randn(I, R)
            V = np.random.randn(J, R)
            W = np.random.randn(K, R)
            
            # ALS iterations
            for it in range(50):
                # Update U
                for r in range(R):
                    # T - sum_{r'!=r} U_r' outer V_r' outer W_r'
                    # This is slow. Better to use matrix unfolding.
                    # T_unfolded_I: shape (I, J*K)
                    # T_unfolded_I[:, :] = T.reshape(I, J*K)
                    # V_unfolded: shape (J*K, R) where V_unfolded[j*K+k, r] = V[j,r]*W[k,r]
                    # U_r = (V_unfolded^T V_unfolded)^-1 V_unfolded^T (T_unfolded_I[:, r])
                    
                    # Precompute unfolded matrices outside the r loop for efficiency?
                    # For simplicity and given small size, we can do it directly.
                    
                    # Residual R_tensor = T - sum_{r'!=r} ...
                    # This is O(R * I * J * K) per update.
                    # I=9, J=18, K=18. I*J*K = 2916.
                    # R=45. 45 * 2916 * 50 iters = ~6.5M ops. Very fast.
                    
                    # Actually, let's just use the direct sum for simplicity in code, 
                    # as numpy operations are fast.
                    
                    pass # Placeholder for logic
                    
            # This implementation is getting complex. 
            # Let's use a simpler heuristic: 
            # Just return the schoolbook rank 54 if we can't find better quickly?
            # No, we want to improve.
            
            # Let's try a specific known decomposition if available?
            # I don't have one.
            
            # Let's try to implement the ALS properly.
            
            T_I = T.reshape(I, J*K)
            T_J = T.transpose(1,0,2).reshape(J, I*K)
            T_K = T.transpose(2,0,1).reshape(K, I*J)
            
            for it in range(100):
                # Update U
                for r in range(R):
                    # Compute V_outer_W for all r
                    # VW = V.T outer W.T ? No.
                    # We need matrix M_r of shape (J*K, R)
                    # M_r[j*K+k, r'] = V[j, r'] * W[k, r']
                    # This is the Khatri-Rao product of V and W.
                    # V shape (J, R), W shape (K, R).
                    # Khatri-Rao product of V and W is shape (J*K, R).
                    # (V ⊙ W)[j*K+k, r] = V[j, r] * W[k, r]
                    
                    # We can compute V_kron_W = np.kron(V, W)? No, that's not right for CP.
                    # It is the column-wise Kronecker product.
                    # V_kron_W = np.zeros((J*K, R))
                    # for r in range(R):
                    #    V_kron_W[:, r] = np.kron(V[:, r], W[:, r])
                    
                    # Let's compute it efficiently.
                    # V_kron_W = (W[:, :, None] * V[None, :, :]).reshape(J*K, R) ?
                    # V[None, :, :] is (1, J, R)
                    # W[:, :, None] is (K, R, 1)
                    # Product is (K, J, R).
                    # We want index j*K+k. So reshape to (J, K, R) then transpose?
                    # V_kron_W = (V[:, None, :] * W[None, :, :]).reshape(J*K, R)
                    # V[:, None, :] is (J, 1, R)
                    # W[None, :, :] is (1, K, R)
                    # Product is (J, K, R).
                    # Reshape to (J*K, R). Index j*K+k. Correct.
                    
                    V_kron_W = (V[:, None, :] * W[None, :, :]).reshape(J*K, R)
                    
                    # Solve U[:, r] = V_kron_W^+ T_I[:, r]
                    # Using least squares
                    # U[:, r] = np.linalg.lstsq(V_kron_W, T_I[:, r], rcond=None)[0]
                    
                    # Update all U at once?
                    # U = V_kron_W^+ T_I
                    # V_kron_W^+ is (J*K, R) ^+ -> (R, J*K)
                    # U = (V_kron_W.T @ V_kron_W)^-1 V_kron_W.T @ T_I
                    try:
                        U = np.linalg.lstsq(V_kron_W, T_I, rcond=None)[0]
                    except:
                        pass
                        
                # Update V
                U_kron_W = (U[:, None, :] * W[None, :, :]).reshape(I*K, R)
                try:
                    V = np.linalg.lstsq(U_kron_W, T_J, rcond=None)[0]
                except:
                    pass
                    
                # Update W
                U_kron_V = (U[:, None, :] * V[None, :, :]).reshape(I*J, R)
                try:
                    W = np.linalg.lstsq(U_kron_V, T_K, rcond=None)[0]
                except:
                    pass
                    
            # Check residual
            # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # Residual = || T - T_approx ||_F
            # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(T - T_approx)
            
            if residual < 1e-6:
                # Try to round
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_approx_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                if np.allclose(T, T_approx_int, atol=1e-9):
                    best_rank = R
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            break

    if best_rank is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
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
