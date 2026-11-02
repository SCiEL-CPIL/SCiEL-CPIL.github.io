import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    time_limit = 110
    
    def is_nice(matrix, tol=1e-8):
        # Check if values are close to integers or half-integers
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                # Try integer
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    res_row.append(int(nearest_int))
                    continue
                # Try half-integer
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_half) < tol:
                    # Store as float or int? Interface accepts integer/half-integer.
                    # If it's an integer, store as int.
                    if nearest_half == int(nearest_half):
                        res_row.append(int(nearest_half))
                    else:
                        res_row.append(nearest_half)
                    continue
                return False, None
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-8:
                        return False
        return True

    # Ranks to try. Lower bound for 3x4x5 is approx 30-35.
    # We prioritize ranks near 35.
    ranks_to_try = [33, 34, 35, 36, 37, 38, 39, 40, 42, 45]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Allocate more restarts for lower ranks
        num_restarts = 300 if R <= 36 else 150
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Sparse random initialization
            # Probability of non-zero: 0.3
            U_rand = [[random.choice([-1, 1]) if random.random() < 0.3 else 0 for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 1]) if random.random() < 0.3 else 0 for _ in range(R)] for _ in range(mp)]
            
            # Build A
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            ok, W_nice = is_nice(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    return {"U": U_rand, "V": V_rand, "W": W_nice}

        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
