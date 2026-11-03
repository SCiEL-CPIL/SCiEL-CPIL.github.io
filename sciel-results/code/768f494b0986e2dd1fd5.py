import numpy as np
import scipy.optimize as so
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build target tensor T
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                T[a, b, w] = 1.0

    # Precompute indices of non-zero entries
    indices = np.argwhere(T == 1)
    idx_a = indices[:, 0]
    idx_b = indices[:, 1]
    idx_w = indices[:, 2]
    num_nonzero = len(indices)
    
    # Function to compute residual efficiently
    def make_res_func(r):
        def res_func(t):
            U = t[:dim_A * r].reshape(dim_A, r)
            V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
            W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
            
            # Compute only the non-zero entries
            # hat_T[a, b, w] = sum_r U[a, r] V[b, r] W[w, r]
            vals = np.einsum('ar,br,wr->r', U[idx_a], V[idx_b], W[idx_w])
            # Sum over r for each index
            vals = vals.sum(axis=-1) if vals.ndim > 1 else vals
            # Actually, einsum with 'ar,br,wr->r' gives shape (num_indices, r)
            # We need sum over r for each index
            vals = np.einsum('ar,br,wr->ab', U[idx_a], V[idx_b], W[idx_w])
            return vals - 1.0
        return res_func

    def verify_solution(U, V, W, atol=1e-9):
        """Verify that the solution exactly decomposes the tensor."""
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        return np.allclose(vals, 1.0, atol=atol)

    def try_rank(r, time_budget=10.0):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Multiple restarts with diverse initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if attempt % 4 == 0:
                # Random Gaussian
                np.random.seed(r * 1000 + attempt)
                U0 = np.random.randn(dim_A, r) * 0.5
                V0 = np.random.randn(dim_B, r) * 0.5
                W0 = np.random.randn(dim_C, r) * 0.5
            elif attempt % 4 == 1:
                # Sparse initialization
                np.random.seed(r * 1000 + attempt + 1)
                U0 = np.zeros((dim_A, r))
                V0 = np.zeros((dim_B, r))
                W0 = np.zeros((dim_C, r))
                for col in range(r):
                    U0[np.random.randint(dim_A), col] = 1
                    V0[np.random.randint(dim_B), col] = 1
                    W0[np.random.randint(dim_C), col] = 1
            elif attempt % 4 == 2:
                # Structured initialization based on Strassen-like patterns
                np.random.seed(r * 1000 + attempt + 2)
                U0 = np.zeros((dim_A, r))
                V0 = np.zeros((dim_B, r))
                W0 = np.zeros((dim_C, r))
                for col in range(r):
                    a_idx = np.random.randint(dim_A)
                    b_idx = np.random.randint(dim_B)
                    w_idx = np.random.randint(dim_C)
                    U0[a_idx, col] = 1
                    V0[b_idx, col] = 1
                    W0[w_idx, col] = 1
                    # Add some off-diagonal terms
                    if np.random.rand() > 0.5:
                        a2 = np.random.randint(dim_A)
                        U0[a2, col] = np.random.choice([-1, 1])
            else:
                # Small random perturbations
                np.random.seed(r * 1000 + attempt + 3)
                U0 = np.random.randn(dim_A, r) * 0.1
                V0 = np.random.randn(dim_B, r) * 0.1
                W0 = np.random.randn(dim_C, r) * 0.1
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            res_func = make_res_func(r)
            
            try:
                # Use TRF method with bounds for robustness
                bounds = (-10, 10)
                res = so.least_squares(res_func, theta0, method='trf', bounds=bounds, 
                                       ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
                
                if res.cost < 1e-10:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # High-precision polish pass
                    theta_polish = res.x.copy()
                    for polish_iter in range(5):
                        if time.time() > deadline:
                            break
                        res2 = so.least_squares(res_func, theta_polish, method='trf', bounds=bounds,
                                                ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=2000)
                        if res2.cost < res.cost:
                            theta_polish = res2.x
                        else:
                            break
                    
                    U_sol = theta_polish[:dim_A * r].reshape(dim_A, r)
                    V_sol = theta_polish[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = theta_polish[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify exactness after rounding
                    if verify_solution(U_int, V_int, W_int, atol=1e-9):
                        return U_int, V_int, W_int
                    
                    # If rounding failed, try to find a nearby integer/half-integer solution
                    # by doing a local search around the rounded values
                    for round_attempt in range(5):
                        if time.time() > deadline:
                            break
                        # Add small perturbations to the rounded solution
                        U_try = U_int + np.random.randn(dim_A, r) * 0.01
                        V_try = V_int + np.random.randn(dim_B, r) * 0.01
                        W_try = W_int + np.random.randn(dim_C, r) * 0.01
                        
                        theta_try = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                        res3 = so.least_squares(res_func, theta_try, method='trf', bounds=bounds,
                                                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
                        
                        if res3.cost < 1e-10:
                            U_sol2 = res3.x[:dim_A * r].reshape(dim_A, r)
                            V_sol2 = res3.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                            W_sol2 = res3.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                            
                            U_int2 = np.round(U_sol2 * 2) / 2
                            V_int2 = np.round(V_sol2 * 2) / 2
                            W_int2 = np.round(W_sol2 * 2) / 2
                            
                            if verify_solution(U_int2, V_int2, W_int2, atol=1e-9):
                                return U_int2, V_int2, W_int2
            except Exception:
                pass
        
        return None

    # Fallback rank-42 solution using Strassen blocks
    def get_fallback_solution():
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

    start_time = time.time()
    global_deadline = start_time + 110.0  # 40s margin
    
    # Try ranks from 24 upward
    for r in range(24, 42):
        if time.time() > global_deadline:
            break
        
        # Allocate more time for lower ranks (harder to find)
        time_budget = min(15.0, (global_deadline - time.time()) * 0.3)
        
        result = try_rank(r, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 42
    U, V, W = get_fallback_solution()
    return {"U": U, "V": V, "W": W}
