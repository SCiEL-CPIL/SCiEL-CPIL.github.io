import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _build_tensor_233():
    n, m, p = 2, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _build_tensor_239():
    n, m, p = 2, 3, 9
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _search_rank_decomposition(T, rank, max_iters=1000, timeout=10.0):
    n_a, n_b, n_c = T.shape
    start_time = time.time()
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() - start_time > timeout:
            break
            
        U = np.random.randn(n_a, rank)
        V = np.random.randn(n_b, rank)
        W = np.random.randn(n_c, rank)
        
        # Normalize to prevent scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Use simple gradient descent or least squares
        # Flatten T
        T_flat = T.flatten()
        
        def loss(params):
            U_p = params[:n_a*rank].reshape(n_a, rank)
            V_p = params[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
            W_p = params[n_a*rank+n_b*rank:].reshape(n_c, rank)
            
            # Compute tensor
            # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is slow in pure python loops, use einsum
            T_comp = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            T_comp_flat = T_comp.flatten()
            err = np.linalg.norm(T_flat - T_comp_flat)
            return err
        
        # Use scipy optimize if available, else simple GD
        try:
            from scipy.optimize import minimize
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = minimize(loss, x0, method='BFGS', options={'maxiter': max_iters})
            if res.success or res.fun < best_err:
                params = res.x
                U_res = params[:n_a*rank].reshape(n_a, rank)
                V_res = params[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
                W_res = params[n_a*rank+n_b*rank:].reshape(n_c, rank)
                
                # Round to half-integers
                U_rounded = np.round(U_res * 2) / 2
                V_rounded = np.round(V_res * 2) / 2
                W_rounded = np.round(W_res * 2) / 2
                
                # Verify
                T_comp = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                err = np.linalg.norm(T_flat - T_comp.flatten())
                
                if err < 1e-6:
                    return U_rounded, V_rounded, W_rounded
                elif err < best_err:
                    best_err = err
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    
        except Exception:
            pass
            
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    # Fallback schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find better decomposition
    # Target: Rank 48 via 3x Rank 16 of M_2,3,3
    T_233 = _build_tensor_233()
    res_233 = _search_rank_decomposition(T_233, 16, timeout=20.0)
    
    if res_233 is not None:
        U3, V3, W3 = res_233
        rank_3 = 16
        rank_total = rank_3 * 3
        
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        for s in range(3):
            for r in range(rank_3):
                col = s * rank_3 + r
                U[:, col] = U3[:, r]
                # V mapping: block s of B (3x3)
                # V index in 239: j*9 + (3s + k)
                # V index in 233: j*3 + k
                for j in range(m):
                    for k in range(3):
                        idx_239 = j * 9 + (3 * s + k)
                        idx_233 = j * 3 + k
                        V[idx_239, col] = V3[idx_233, r]
                        
                # W mapping: block s of C (2x3)
                # W index in 239: (3s + k)*2 + i
                # W index in 233: k*2 + i
                for i in range(n):
                    for k in range(3):
                        idx_239 = (3 * s + k) * 2 + i
                        idx_233 = k * 2 + i
                        W[idx_239, col] = W3[idx_233, r]
                        
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": U_school,
            "V": V_school,
            "W": W_school
        }
