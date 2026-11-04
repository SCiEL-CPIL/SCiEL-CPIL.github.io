import numpy as np
import random
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    start_time = time.time()
    deadline = start_time + 140

    def compute_error_np(U_int, V_int, W_int, r):
        U_t = np.array(U_int).reshape(n, m, r)
        V_t = np.array(V_int).reshape(m, p, r)
        W_t = np.array(W_int).reshape(p, n, r)
        
        S = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] = np.dot(U_t[i, j, :], V_t[j, k, :] * W_t[k, i, :])
        return np.sum((S - 8) ** 2)

    def try_find_decomposition(r):
        U_int = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(n * m)]
        V_int = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(m * p)]
        W_int = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(p * n)]
        
        current_error = compute_error_np(U_int, V_int, W_int, r)
        
        t = 1.0
        for step in range(2000):
            if time.time() > deadline:
                return None
            
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                rows = n * m
            elif mat_idx == 1:
                rows = m * p
            else:
                rows = p * n
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, r - 1)
            
            old_val = U_int[row][col] if mat_idx == 0 else (V_int[row][col] if mat_idx == 1 else W_int[row][col])
            delta = random.choice([-2, -1, 0, 1, 2])
            new_val = old_val + delta
            
            if mat_idx == 0:
                U_int[row][col] = new_val
            elif mat_idx == 1:
                V_int[row][col] = new_val
            else:
                W_int[row][col] = new_val
            
            new_error = compute_error_np(U_int, V_int, W_int, r)
            
            if new_error < current_error:
                current_error = new_error
            else:
                if mat_idx == 0:
                    U_int[row][col] = old_val
                elif mat_idx == 1:
                    V_int[row][col] = old_val
                else:
                    W_int[row][col] = old_val
            
            if new_error >= current_error:
                delta_e = new_error - current_error
                if random.random() < np.exp(-delta_e / t):
                    pass
                else:
                    if mat_idx == 0:
                        U_int[row][col] = old_val
                    elif mat_idx == 1:
                        V_int[row][col] = old_val
                    else:
                        W_int[row][col] = old_val
            
            t *= 0.99
            
            if current_error == 0:
                U_half = [[val / 2 for val in row] for row in U_int]
                V_half = [[val / 2 for val in row] for row in V_int]
                W_half = [[val / 2 for val in row] for row in W_int]
                return {"U": U_half, "V": V_half, "W": W_half}
        
        return None

    for r in range(50, 15, -1):
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            result = try_find_decomposition(r)
            if result is not None:
                U = result["U"]
                V = result["V"]
                W = result["W"]
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_result = result
                    best_rank = r
                    return best_result
                    
    return best_result
