import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor T indices
    # T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    # Convert to numpy arrays for fast indexing
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    num_terms = len(indices)  # 48
    
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
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
        
        # Block 1: k=0 (b cols 0-5), i=0 (w rows 0,3,6,9) -> w indices 0, 2, 1, 3
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        # Block 2: k=1 (b cols 6-11), i=0 (w rows 1,4,7,10) -> w indices 4, 6, 5, 7
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        # Block 3: k=2 (b cols 12-17), i=0 (w rows 2,5,8,11) -> w indices 8, 10, 9, 11
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback()
    
    # Verify fallback
    U_fb_arr = np.array(U_fb, dtype=np.float64)
    V_fb_arr = np.array(V_fb, dtype=np.float64)
    W_fb_arr = np.array(W_fb, dtype=np.float64)
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        
        # Compute hat_T at indices
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        
        # Check that all target entries are 1
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        # Check that non-target entries are 0 (approximately)
        full_hat = hat_T
        full_hat[idx_a, idx_b, idx_w] = 0
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9
    
    # Verify fallback
    if not verify_solution(U_fb_arr, V_fb_arr, W_fb_arr):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 42
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def residual_sparse(theta, r):
        """Compute residual only on the 48 target entries (much faster)"""
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        # Compute U * V^T element-wise with W for each target index
        # hat_T[a,b,w] = sum_r U[a,r] * V[b,r] * W[w,r]
        # We only need the 48 target entries
        res = np.zeros(num_terms)
        for t in range(num_terms):
            a, b, w = idx_a[t], idx_b[t], idx_w[t]
            res[t] = np.sum(U[a] * V[b] * W[w]) - 1.0
        return res
    
    def try_rank_optimization(r, num_restarts=10):
        """Try to find a rank-r decomposition using optimization on sparse residual"""
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Structured initialization
            np.random.seed(r * 1000 + attempt)
            U0 = np.zeros((dim_A, r))
            V0 = np.zeros((dim_B, r))
            W0 = np.zeros((dim_C, r))
            
            # Initialize with sparse random pattern
            for t in range(r):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                a = i * m + j
                b = j * p + k
                w = k * n + i
                
                U0[a, t] = 1.0 + np.random.randn() * 0.5
                V0[b, t] = 1.0 + np.random.randn() * 0.5
                W0[w, t] = 1.0 + np.random.randn() * 0.5
                
                # Add some cross terms
                if r > 1 and np.random.rand() < 0.5:
                    i2 = np.random.randint(n)
                    j2 = np.random.randint(m)
                    k2 = np.random.randint(p)
                    a2 = i2 * m + j2
                    b2 = j2 * p + k2
                    w2 = k2 * n + i2
                    U0[a2, t] += np.random.randn() * 0.3
                    V0[b2, t] += np.random.randn() * 0.3
                    W0[w2, t] += np.random.randn() * 0.3
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                import scipy.optimize as so
                
                # Use BFGS with L-BFGS-B as fallback
                res = so.minimize(
                    lambda t: np.sum(residual_sparse(t, r)**2),
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.fun < 1e-10:
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
                        U_int2 = np.round(U_sol)
                        V_int2 = np.round(V_sol)
                        W_int2 = np.round(W_sol)
                        if verify_solution(U_int2, V_int2, W_int2):
                            return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
            except Exception:
                pass
        
        return None
    
    # Try ranks from 24 to 41
    for r in range(24, 42):
        if time.time() > deadline:
            break
        
        res = try_rank_optimization(r, num_restarts=8)
        if res:
            best_rank = r
            best_result = res
            break
    
    return best_result
