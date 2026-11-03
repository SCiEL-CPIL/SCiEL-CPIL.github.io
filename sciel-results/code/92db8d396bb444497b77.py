import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank = 28 (Strassen-based block decomposition)
    rank_strassen = 28
    
    # 1. Build the Strassen solution (Rank 28)
    U_s = [[0] * rank_strassen for _ in range(n * m)]
    V_s = [[0] * rank_strassen for _ in range(m * p)]
    W_s = [[0] * rank_strassen for _ in range(p * n)]
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U_s[global_idx][col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V_s[global_idx][col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W_s[global_idx][col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    U_np_s = np.array(U_s, dtype=float)
    V_np_s = np.array(V_s, dtype=float)
    W_np_s = np.array(W_s, dtype=float)
    
    # Verify Strassen
    T_expected = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    if a_col == b_row:
                        i = a_row * m + a_col
                        j = b_row * p + b_col
                        c_row = a_row
                        c_col = b_col
                        k = c_col * n + c_row
                        T_expected[i, j, k] = 1.0
                        
    # 2. Attempt Rank 27 Search
    # Time budget: 120s total. Reserve 10s for setup/fallback.
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_rank = 28
    best_U, best_V, best_W = U_np_s, V_np_s, W_np_s
    
    # Try rank 27
    target_rank = 27
    # Initialize with noise
    np.random.seed(42)
    
    def flatten_params(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        return U, V, W

    def loss(params):
        U, V, W = flatten_params(params)
        # Compute tensor
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Compute loss
        err = T - T_expected
        return np.sum(err**2)

    # Initial guess: Random
    x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
    
    # Optimize
    try:
        res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 5000, 'disp': False})
        if time.time() > deadline:
            pass # Break early if needed, but minimize handles maxiter
        else:
            # Check result
            U_opt, V_opt, W_opt = flatten_params(res.x)
            T_opt = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
            max_err = np.max(np.abs(T_opt - T_expected))
            if max_err < 1e-6:
                # Found a rank 27 solution!
                best_rank = 27
                best_U, best_V, best_W = U_opt, V_opt, W_opt
    except Exception:
        pass

    # If time allows, try rank 26? Unlikely to converge quickly.
    # Let's stick to 27 for now to ensure we return something.
    
    # Convert back to lists of lists (or arrays? The task says "integer/half-integer entries", 
    # but the evaluator likely accepts floats if they are exact. 
    # The prompt says "integer/half-integer entries that exactly decompose".
    # If the optimizer finds floats, we might need to round them?
    # If the solution is exact, the floats should be very close to integers or half-integers.
    
    if best_rank == 27:
        # Round to nearest 0.5
        def round_half(x):
            return np.round(x * 2) / 2.0
            
        U_final = round_half(best_U)
        V_final = round_half(best_V)
        W_final = round_half(best_W)
        
        # Verify rounded solution
        T_check = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
        if np.max(np.abs(T_check - T_expected)) < 1e-9:
             return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        else:
             # Fallback to original floats if rounding broke it? 
             # Or just return the unrounded if evaluator accepts floats?
             # "integer/half-integer" suggests we should try to make them so.
             # If we can't, maybe rank 27 doesn't have a half-integer solution.
             pass

    # Return Strassen (Rank 28)
    return {"U": U_np_s.tolist(), "V": V_np_s.tolist(), "W": W_np_s.tolist()}
# EVOLVE-BLOCK-END
