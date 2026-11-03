import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = start_time + 110.0
    
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    rank_terms = []
    
    for half in range(2):
        k_offset = 3 * half
        
        # 4x4x2 part
        for i1 in range(2):
            for j1 in range(2):
                for s in range(7):
                    u_s, v_s, w_s = strassen_terms_2x2[s]
                    U_global = {}
                    V_global = {}
                    W_global = {}
                    
                    for (i2, j2), cu in u_s.items():
                        gi = 2 * i1 + i2
                        gj = 2 * j1 + j2
                        key = (gi, gj)
                        U_global[key] = U_global.get(key, 0) + cu
                        
                    for (j2, k2), cv in v_s.items():
                        gj = 2 * j1 + j2
                        gk = k_offset + k2
                        key = (gj, gk)
                        V_global[key] = V_global.get(key, 0) + cv
                        
                    for (i2, k2), cw in w_s.items():
                        gi = 2 * i1 + i2
                        gk = k_offset + k2
                        key = (gk, gi)
                        W_global[key] = W_global.get(key, 0) + cw
                        
                    rank_terms.append((U_global, V_global, W_global))
                    
        # 4x4x1 part (rank 4)
        k1 = k_offset + 2
        for r in range(4):
            U_global = {}
            V_global = {}
            W_global = {}
            for i in range(4):
                if i == r:
                    for j in range(4):
                        U_global[(i, j)] = 1
            for j in range(4):
                V_global[(j, k1)] = 1
            W_global[(k1, r)] = 1
            rank_terms.append((U_global, V_global, W_global))
    
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
