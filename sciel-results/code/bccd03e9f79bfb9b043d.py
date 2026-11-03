import numpy as np
import scipy.optimize as so
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build target tensor T indices
    # T is a sparse tensor: T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
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
    
    def residual(theta, r):
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        # Compute full tensor for better gradient flow
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Residual on sparse indices (where T=1)
        vals = hat_T[idx_a, idx_b, idx_w]
        res_sparse = vals - 1.0
        
        # Also penalize non-zero entries where T should be 0
        # Build a mask for where T is zero
        full_T = np.zeros((dim_A, dim_B, dim_C))
        full_T[idx_a, idx_b, idx_w] = 1.0
        zero_mask = (full_T == 0)
        
        # Get values at zero positions
        zero_vals = hat_T[zero_mask]
        res_zero = zero_vals
        
        # Combine: sparse residual weighted more heavily
        return np.concatenate([res_sparse, 0.1 * res_zero])
    
    best_result = None
    best_rank = 42
    best_U, best_V, best_W = None, None, None
    
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
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Search for lower rank
    for r in range(24, 42):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            # Random init with varying scales
            np.random.seed(r * 100 + attempt)
            scale = 1.0 + 0.5 * (attempt % 4)
            U0 = np.random.randn(dim_A, r) * scale
            V0 = np.random.randn(dim_B, r) * scale
            W0 = np.random.randn(dim_C, r) * scale
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def res_func(t):
                return residual(t, r)
            
            try:
                # First pass: find a good solution
                res = so.least_squares(res_func, theta0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if not res.success:
                    continue
                    
                U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                
                # Check if we're close enough to try rounding
                if res.cost < 1e-6:
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify exactness on sparse indices
                    hat_T_int = np.einsum('ar,br,wr->abw', U_int, V_int, W_int)
                    vals_int = hat_T_int[idx_a, idx_b, idx_w]
                    
                    if np.allclose(vals_int, 1.0, atol=1e-9):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    else:
                        # Rounding failed. Try local polish: adjust nearby half-integer values
                        # For each matrix, try flipping some entries to adjacent half-integers
                        best_local_cost = res.cost
                        best_local_U, best_local_V, best_local_W = U_int.copy(), V_int.copy(), W_int.copy()
                        
                        # Try small perturbations: flip individual entries by +/-0.5
                        for _ in range(50):  # limited local search steps
                            if time.time() > deadline:
                                break
                            
                            # Pick a random matrix and entry to flip
                            mat_idx = np.random.randint(3)
                            if mat_idx == 0:
                                row, col = np.random.randint(dim_A), np.random.randint(r)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                U_int[row, col] += delta
                                U_check = U_int.copy()
                            elif mat_idx == 1:
                                row, col = np.random.randint(dim_B), np.random.randint(r)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                V_int[row, col] += delta
                                V_check = V_int.copy()
                            else:
                                row, col = np.random.randint(dim_C), np.random.randint(r)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                W_int[row, col] += delta
                                W_check = W_int.copy()
                            
                            # Evaluate
                            hat_T_test = np.einsum('ar,br,wr->abw', U_int, V_int, W_int)
                            vals_test = hat_T_test[idx_a, idx_b, idx_w]
                            test_cost = np.sum((vals_test - 1.0)**2)
                            
                            if test_cost < best_local_cost:
                                best_local_cost = test_cost
                                best_local_U, best_local_V, best_local_W = U_int.copy(), V_int.copy(), W_int.copy()
                            else:
                                # Revert
                                if mat_idx == 0:
                                    U_int[row, col] -= delta
                                elif mat_idx == 1:
                                    V_int[row, col] -= delta
                                else:
                                    W_int[row, col] -= delta
                        
                        U_int, V_int, W_int = best_local_U, best_local_V, best_local_W
                        
                        # Final verification
                        hat_T_int = np.einsum('ar,br,wr->abw', U_int, V_int, W_int)
                        vals_int = hat_T_int[idx_a, idx_b, idx_w]
                        if np.allclose(vals_int, 1.0, atol=1e-9):
                            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception as e:
                pass
                
    # If no better rank found, return the known rank 42 solution
    return {"U": U_fb, "V": V_fb, "W": W_fb}


# Pre-compute the zero mask for efficiency
def _precompute_zero_mask():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    full_T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                full_T[a, b, w] = 1.0
    
    return full_T == 0

_ZERO_MASK = _precompute_zero_mask()
