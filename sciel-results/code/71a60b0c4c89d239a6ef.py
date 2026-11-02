import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = 3, 3, 6
    
    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = generate_schoolbook()
    best_rank = n * m * p
    
    def find_3x3x2_decomposition(R, max_attempts=500):
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization with wider range
            U_r = np.random.randint(-3, 4, size=(9, R))
            V_r = np.random.randint(-3, 4, size=(6, R))
            W_r = np.zeros((6, R))
            
            valid = True
            for i in range(3):
                for k in range(2):
                    M = np.zeros((9, R))
                    for j in range(3):
                        for jp in range(3):
                            M[j * 3 + jp, :] = U_r[i * 3 + j, :] * V_r[jp * 2 + k, :]
                    
                    rhs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
                    
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-6:
                        W_r[i * 2 + k, :] = w_vec
                    else:
                        valid = False
                        break
            
            if valid:
                U_np = U_r.astype(float)
                V_np = V_r.astype(float)
                W_np = W_r.astype(float)
                
                correct = True
                for i in range(3):
                    for ip in range(3):
                        for k in range(2):
                            for kp in range(2):
                                for j in range(3):
                                    for jp in range(3):
                                        expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                        actual = 0.0
                                        for r in range(R):
                                            actual += U_np[i * 3 + j, r] * V_np[jp * 2 + k, r] * W_np[ip * 2 + kp, r]
                                        if abs(actual - expected) > 1e-6:
                                            correct = False
                                            break
                                    if not correct: break
                                if not correct: break
                            if not correct: break
                        if not correct: break
                    if not correct: break
                if correct:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    for R in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
        decomp = find_3x3x2_decomposition(R)
        if decomp is not None:
            U_core, V_core, W_core = decomp
            total_rank = 3 * R
            
            if total_rank < best_rank:
                U_tot = [[0] * total_rank for _ in range(n * m)]
                V_tot = [[0] * total_rank for _ in range(m * p)]
                W_tot = [[0] * total_rank for _ in range(p * n)]
                
                for block in range(3):
                    offset = block * R
                    for i in range(3):
                        for j in range(3):
                            for r in range(R):
                                U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
                    
                    for j in range(3):
                        for k in range(2):
                            for r in range(R):
                                V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
                    
                    for i in range(3):
                        for k in range(2):
                            for r in range(R):
                                W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
                
                best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                best_rank = total_rank
                break
    
    return best_result
