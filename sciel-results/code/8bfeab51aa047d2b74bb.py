import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    rank = 98
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    products = [
        {
            "X": [(11, 1), (22, 1)],
            "Y": [(11, 1), (22, 1)],
            "C": {11: 1, 22: 1}
        },
        {
            "X": [(21, 1), (22, 1)],
            "Y": [(11, 1)],
            "C": {21: 1, 22: -1}
        },
        {
            "X": [(11, 1)],
            "Y": [(12, 1), (22, -1)],
            "C": {12: 1, 22: 1}
        },
        {
            "X": [(22, 1)],
            "Y": [(21, 1), (11, -1)],
            "C": {11: 1, 21: 1}
        },
        {
            "X": [(11, 1), (12, 1)],
            "Y": [(22, 1)],
            "C": {11: -1, 12: 1}
        },
        {
            "X": [(21, 1), (11, -1)],
            "Y": [(11, 1), (12, 1)],
            "C": {22: 1}
        },
        {
            "X": [(12, 1), (22, -1)],
            "Y": [(21, 1), (22, 1)],
            "C": {11: 1}
        }
    ]
    
    def strassen_2x2_2x2():
        terms = []
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[0] = 1; u[3] = 1; v[0] = 1; v[3] = 1; w[0] = 1; w[3] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[2] = 1; u[3] = 1; v[0] = 1; w[2] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[0] = 1; v[1] = 1; v[3] = -1; w[1] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[3] = 1; v[2] = 1; v[0] = -1; w[2] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[0] = 1; u[1] = 1; v[3] = 1; w[1] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[2] = 1; u[0] = -1; v[0] = 1; v[1] = 1; w[3] = 1
        terms.append((u, v, w))
        
        u = [0, 0, 0, 0]; v = [0, 0, 0, 0]; w = [0, 0, 0, 0]
        u[1] = 1; u[3] = -1; v[2] = 1; v[3] = 1; w[0] = 1
        terms.append((u, v, w))
        
        return terms

    strassen_terms = strassen_2x2_2x2()
    
    def A_block_indices(block_id, i, j):
        if block_id == 11: return i, j
        elif block_id == 12: return i, j + 2
        elif block_id == 21: return i + 2, j
        elif block_id == 22: return i + 2, j + 2
    
    def B_block_indices(block_id, j, k):
        if block_id == 11: return j, k
        elif block_id == 12: return j, k + 4
        elif block_id == 21: return j + 2, k
        elif block_id == 22: return j + 2, k + 4
    
    def C_block_indices(block_id, k, i):
        if block_id == 11: return k, i
        elif block_id == 12: return k + 4, i
        elif block_id == 21: return k, i + 2
        elif block_id == 22: return k + 4, i + 2

    rank_idx = 0
    for prod in products:
        X_terms = prod["X"]
        Y_terms = prod["Y"]
        C_coeffs = prod["C"]
        
        for a_id, c_a in X_terms:
            for b_id, c_b in Y_terms:
                for half in range(2):
                    for t_idx in range(7):
                        u_local, v_local, w_local = strassen_terms[t_idx]
                        
                        for i in range(2):
                            for j in range(2):
                                if u_local[i * 2 + j] != 0:
                                    I, J = A_block_indices(a_id, i, j)
                                    U[I * 4 + J, rank_idx] += c_a * u_local[i * 2 + j]
                        
                        for j in range(2):
                            for k in range(2):
                                if v_local[j * 2 + k] != 0:
                                    k_global_local = k if half == 0 else k + 2
                                    J, K = B_block_indices(b_id, j, k_global_local)
                                    V[J * 8 + K, rank_idx] += c_b * v_local[j * 2 + k]
                        
                        for k in range(2):
                            for i in range(2):
                                if w_local[k * 2 + i] != 0:
                                    k_global_local = k if half == 0 else k + 2
                                    for c_id, gamma in C_coeffs.items():
                                        K, I = C_block_indices(c_id, k_global_local, i)
                                        W[K * 4 + I, rank_idx] += gamma * w_local[k * 2 + i]
                        
                        rank_idx += 1
                        if time.time() > deadline:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
