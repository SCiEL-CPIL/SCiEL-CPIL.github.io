import numpy as np
import time
import sys
from fractions import Fraction
import scipy.optimize as so

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    num_terms = len(indices)  # 48
    
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Fallback solution (Rank 42) using Strassen
    def get_fallback():
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback()
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Check target entries
        vals = hat_T[idx_a, idx_b, idx_w]
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        # Check non-target entries
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9
    
    if not verify_solution(U_fb, V_fb, W_fb):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 42
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def residual_full(theta, r):
        """Compute full residual sum of squares"""
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Residual on target indices
        res_target = hat_T[idx_a, idx_b, idx_w] - 1.0
        
        # Residual on non-target indices
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        res_non_target = full_hat.flatten()
        
        return np.sum(res_target**2) + np.sum(res_non_target**2)

    def try_rank_optimization(r, num_restarts=5):
        """Try to find a rank-r decomposition using optimization on full residual"""
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(r * 1000 + attempt)
            U0 = np.random.randn(dim_A, r)
            V0 = np.random.randn(dim_B, r)
            W0 = np.random.randn(dim_C, r)
            
            # Scale to reasonable magnitude
            U0 *= 0.5
            V0 *= 0.5
            W0 *= 0.5
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = so.minimize(
                    lambda t: residual_full(t, r),
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res.fun < 1e-8:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_int, V_int, W_int):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    else:
                        # Try rounding to integers
                        U_int2 = np.round(U_sol)
                        V_int2 = np.round(V_sol)
                        W_int2 = np.round(W_sol)
                        if verify_solution(U_int2, V_int2, W_int2):
                            return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
            except Exception:
                pass
        
        return None
    
    # Try ranks from 41 down to 24
    for r in range(41, 23, -1):
        if time.time() > deadline:
            break
        
        res = try_rank_optimization(r, num_restarts=10)
        if res:
            best_rank = r
            best_result = res
            break
    
    return best_result
