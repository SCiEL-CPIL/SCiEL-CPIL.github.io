"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    
    # Build target tensor T
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook solution as fallback
    schoolbook_rank = n * m * p  # 42
    U_sb = [[0] * schoolbook_rank for _ in range(N_U)]
    V_sb = [[0] * schoolbook_rank for _ in range(N_V)]
    W_sb = [[0] * schoolbook_rank for _ in range(N_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def verify_solution(U_arr, V_arr, W_arr):
        R = U_arr.shape[1]
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc.flatten(), T_flat, atol=1e-8)
    
    def make_residual_func(R):
        len_U = N_U * R
        len_V = N_V * R
        def res(params):
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat
        return res
    
    def try_rank(R, time_budget, num_restarts=50):
        if R >= best_rank:
            return None
        
        res_func = make_residual_func(R)
        total_vars = (N_U + N_V + N_W) * R
        
        # Create structured initializations based on problem structure
        init_list = []
        
        # Strategy 1: Random uniform
        for _ in range(10):
            init_list.append(np.random.uniform(-1, 1, total_vars))
        
        # Strategy 2: Sparse random (many zeros)
        for _ in range(10):
            x = np.random.uniform(-1, 1, total_vars)
            mask = np.random.random(total_vars) < 0.4
            x[~mask] = 0
            init_list.append(x)
        
        # Strategy 3: Based on schoolbook structure with grouping
        for grouping in [1, 2, 3]:
            x = np.zeros(total_vars)
            for j in range(m):
                for g in range(grouping):
                    r_idx = j * grouping + g
                    if r_idx >= R:
                        continue
                    for i in range(n):
                        for k in range(p):
                            if (i + k) % grouping == g:
                                alpha = i * m + j
                                beta = j * p + k
                                gamma = k * n + i
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
            init_list.append(x)
        
        # Strategy 4: Dissection method using Strassen-like structure
        # For 2x3x7, we can exploit the small n=2 dimension
        # Use tensor product of 2x2x2 (Strassen rank 7) and 1x3x7 (rank 3)
        # This gives rank 7*3 = 21, but we need to adapt for rectangular case
        
        # Create a structured initialization based on dissection
        # The idea: group the k-dimension into blocks and use Strassen-like formulas
        for block_size in [2, 3]:
            num_blocks = (p + block_size - 1) // block_size
            x = np.zeros(total_vars)
            
            for b in range(num_blocks):
                k_start = b * block_size
                k_end = min((b + 1) * block_size, p)
                block_p = k_end - k_start
                
                # For each j, create block-specific terms
                for j in range(m):
                    # Use Strassen-like 7 terms for the 2x2xblock_p subproblem
                    # Simplified: use block_p terms for this block
                    for local_k in range(block_p):
                        r_idx = b * m * block_p + j * block_p + local_k
                        if r_idx >= R:
                            continue
                        k = k_start + local_k
                        for i in range(n):
                            alpha = i * m + j
                            beta = j * p + k
                            gamma = k * n + i
                            x[alpha * R + r_idx] = 1.0
                            x[(N_U + beta) * R + r_idx] = 1.0
                            x[(N_U + N_V + gamma) * R + r_idx] = 1.0
            init_list.append(x)
        
        # Strategy 5: Use known rank 40 decomposition structure
        # R(2,3,7) = 40 is known; try to find it
        # Based on dissection: 2x3x7 = 2x3x2 + 2x3x2 + 2x3x2 + 2x3x1
        # R(2,3,2) = 7, R(2,3,1) = 4 (or less)
        # Total: 3*7 + 4 = 25 (theoretical), but practical might be higher
        
        # Create initialization based on this dissection
        x = np.zeros(total_vars)
        # First 3 blocks of size 2: use 7 terms each = 21 terms
        # Last block of size 1: use 4 terms = 4 terms
        # Total: 25 terms (if R >= 25)
        
        if R >= 25:
            # Block 1: k=0,1
            for j in range(m):
                for t in range(7):
                    r_idx = t
                    if r_idx >= R:
                        continue
                    # Strassen-like terms for 2x3x2
                    # Simplified: just set up basic structure
                    for i in range(n):
                        for k in range(2):
                            alpha = i * m + j
                            beta = j * p + k
                            gamma = k * n + i
                            # Use different values for different (i,k) to allow cancellation
                            if t < 4:  # First 4 terms for basic multiplication
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
                            else:  # Additional terms for Strassen-like optimization
                                # These will be optimized to reduce rank
                                x[alpha * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + beta) * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + N_V + gamma) * R + r_idx] = np.random.uniform(-1, 1)
            
            # Block 2: k=2,3
            for j in range(m):
                for t in range(7):
                    r_idx = 21 + j * 7 + t
                    if r_idx >= R:
                        continue
                    for i in range(n):
                        for k in range(2, 4):
                            alpha = i * m + j
                            beta = j * p + k
                            gamma = k * n + i
                            if t < 4:
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
                            else:
                                x[alpha * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + beta) * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + N_V + gamma) * R + r_idx] = np.random.uniform(-1, 1)
            
            # Block 3: k=4,5
            for j in range(m):
                for t in range(7):
                    r_idx = 21 + m * 7 + j * 7 + t
                    if r_idx >= R:
                        continue
                    for i in range(n):
                        for k in range(4, 6):
                            alpha = i * m + j
                            beta = j * p + k
                            gamma = k * n + i
                            if t < 4:
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
                            else:
                                x[alpha * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + beta) * R + r_idx] = np.random.uniform(-1, 1)
                                x[(N_U + N_V + gamma) * R + r_idx] = np.random.uniform(-1, 1)
            
            # Block 4: k=6 (single element)
            for j in range(m):
                for t in range(4):
                    r_idx = 21 + 2 * m * 7 + j * 4 + t
                    if r_idx >= R:
                        continue
                for i in range(n):
                    alpha = i * m + j
                    beta = j * p + 6
                    gamma = 6 * n + i
                    x[alpha * R + 21 + 2 * m * 7 + j * 4] = 1.0
                    x[(N_U + beta) * R + 21 + 2 * m * 7 + j * 4] = 1.0
                    x[(N_U + N_V + gamma) * R + 21 + 2 * m * 7 + j * 4] = 1.0
            
            init_list.append(x)
        
        best_sol = None
        best_cost = np.inf
        
        end_time = time.time() + time_budget
        
        from scipy.optimize import least_squares
        
        for x0 in init_list:
            if time.time() > end_time:
                break
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        # If we have time, try more random restarts
        while time.time() < end_time and best_cost > 1e-4:
            x0 = np.random.uniform(-2, 2, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=300)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None or best_cost > 1e-3:
            return None
        
        # Try rounding to half-integers
        x_rounded = np.round(best_sol * 2) / 2.0
        
        U_r = x_rounded[:N_U * R].reshape(N_U, R)
        V_r = x_rounded[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
        W_r = x_rounded[N_U * R + N_V * R:].reshape(N_W, R)
        
        if verify_solution(U_r, V_r, W_r):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        
        # Try rounding to integers
        x_rounded_int = np.round(best_sol)
        U_r = x_rounded_int[:N_U * R].reshape(N_U, R)
        V_r = x_rounded_int[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
        W_r = x_rounded_int[N_U * R + N_V * R:].reshape(N_W, R)
        
        if verify_solution(U_r, V_r, W_r):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        
        # Try the unrounded solution if it's very close
        if best_cost < 1e-10:
            U_r = best_sol[:N_U * R].reshape(N_U, R)
            V_r = best_sol[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
            W_r = best_sol[N_U * R + N_V * R:].reshape(N_W, R)
            if verify_solution(U_r, V_r, W_r):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        return None
    
    # Try ranks from schoolbook-1 down to 25
    ranks_to_try = list(range(schoolbook_rank - 1, 24, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        # Allocate time: more time for higher ranks (closer to schoolbook)
        if R > 35:
            time_budget = min(20, remaining * 0.5)
        elif R > 30:
            time_budget = min(25, remaining * 0.4)
        else:
            time_budget = min(30, remaining * 0.3)
        
        result = try_rank(R, time_budget)
        if result is not None:
            best_result = result
            best_rank = R
            # Continue searching for even lower ranks
            ranks_to_try = [r for r in ranks_to_try if r < R]
    
    return best_result
# EVOLVE-BLOCK-END
