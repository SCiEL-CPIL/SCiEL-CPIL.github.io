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
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    def verify_decomposition(U, V, W, rank):
        # Exact verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def is_half_integer(matrix, tol=1e-7):
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return True, result

    # Search for minimal rank R
    # Lower bound is likely around 24.
    for R in range(24, 46):
        if time.time() - start_time > time_limit:
            break
            
        num_restarts = 5000 if R < 30 else 2000
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random integer vectors for U and V
            U_rand = np.random.choice([-1, 0, 1], size=(nm, R))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R))
            
            # Build linear system for W
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        base_w_idx = (k * n + i) * R
                        coeffs = U_rand[i * m + j] * V_rand[j * p + k]
                        A[eq_idx, base_w_idx : base_w_idx + R] = coeffs
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                if verify_decomposition(U_list, V_list, W_nice, R):
                    return {"U": U_list, "V": V_list, "W": W_nice}

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
