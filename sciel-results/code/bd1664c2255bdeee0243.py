import numpy as np
import random
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_rank = 25
    best_factors = None
    
    for rank in range(20, 25):
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            random.seed(trial * 1000 + rank)
            np.random.seed(trial * 1000 + rank)
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(500):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                for a in range(n*m):
                    for r in range(rank):
                        A_vec = np.zeros(m*p * p*n)
                        b_vec = np.zeros(m*p * p*n)
                        idx = 0
                        for b in range(m*p):
                            for c in range(p*n):
                                A_vec[idx] = V[b,r] * W[c,r]
                                b_vec[idx] = T[a,b,c]
                                for r2 in range(rank):
                                    if r2 != r:
                                        b_vec[idx] -= U[a,r2] * V[b,r2] * W[c,r2]
                                idx += 1
                        denom = np.dot(A_vec, A_vec)
                        if denom > 1e-12:
                            U[a,r] = np.dot(A_vec, b_vec) / denom
                
                # Update V
                for b in range(m*p):
                    for r in range(rank):
                        A_vec = np.zeros(n*m * p*n)
                        b_vec = np.zeros(n*m * p*n)
                        idx = 0
                        for a in range(n*m):
                            for c in range(p*n):
                                A_vec[idx] = U[a,r] * W[c,r]
                                b_vec[idx] = T[a,b,c]
                                for r2 in range(rank):
                                    if r2 != r:
                                        b_vec[idx] -= U[a,r2] * V[b,r2] * W[c,r2]
                                idx += 1
                        denom = np.dot(A_vec, A_vec)
                        if denom > 1e-12:
                            V[b,r] = np.dot(A_vec, b_vec) / denom
                
                # Update W
                for c in range(p*n):
                    for r in range(rank):
                        A_vec = np.zeros(n*m * m*p)
                        b_vec = np.zeros(n*m * m*p)
                        idx = 0
                        for a in range(n*m):
                            for b in range(m*p):
                                A_vec[idx] = U[a,r] * V[b,r]
                                b_vec[idx] = T[a,b,c]
                                for r2 in range(rank):
                                    if r2 != r:
                                        b_vec[idx] -= U[a,r2] * V[b,r2] * W[c,r2]
                                idx += 1
                        denom = np.dot(A_vec, A_vec)
                        if denom > 1e-12:
                            W[c,r] = np.dot(A_vec, b_vec) / denom
                
                # Check error
                err = 0.0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = 0.0
                            for r in range(rank):
                                val += U[a,r] * V[b,r] * W[c,r]
                            err += (val - T[a,b,c])**2
                if err < 1e-8:
                    best_rank = rank
                    best_factors = (U.copy(), V.copy(), W.copy())
                    break
            
            if best_rank == rank:
                break
        if best_rank < 25:
            break
    
    if best_factors is not None:
        U, V, W = best_factors
        # Convert to half-integers
        U_out = [[0.0]*rank for _ in range(n*m)]
        V_out = [[0.0]*rank for _ in range(m*p)]
        W_out = [[0.0]*rank for _ in range(p*n)]
        for i in range(n*m):
            for r in range(rank):
                U_out[i][r] = round(U[i,r]*2)/2
        for i in range(m*p):
            for r in range(rank):
                V_out[i][r] = round(V[i,r]*2)/2
        for i in range(p*n):
            for r in range(rank):
                W_out[i][r] = round(W[i,r]*2)/2
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to rank 25
        rank = 25
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        strassen_V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        for b in range(3):
            for r in range(7):
                for i in range(4):
                    U[i][col] += strassen_U[r][i]
                for loc_idx, coeff in strassen_V_local[r].items():
                    if loc_idx == 0: v_idx = 0 * p + (2 * b)
                    elif loc_idx == 1: v_idx = 0 * p + (2 * b + 1)
                    elif loc_idx == 2: v_idx = 1 * p + (2 * b)
                    elif loc_idx == 3: v_idx = 1 * p + (2 * b + 1)
                    V[v_idx][col] += coeff
                for loc_idx, coeff in strassen_W_local[r].items():
                    if loc_idx == 0: w_idx = (2 * b) * n + 0
                    elif loc_idx == 1: w_idx = (2 * b + 1) * n + 0
                    elif loc_idx == 2: w_idx = (2 * b) * n + 1
                    elif loc_idx == 3: w_idx = (2 * b + 1) * n + 1
                    W[w_idx][col] += coeff
                col += 1
        schoolbook_U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        schoolbook_V_local = [{0:1},{1:1},{0:1},{1:1}]
        schoolbook_W_local = [{0:1},{0:1},{1:1},{1:1}]
        for r in range(4):
            for i in range(4):
                U[i][col] += schoolbook_U[r][i]
            for loc_idx, coeff in schoolbook_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + 6
                elif loc_idx == 1: v_idx = 1 * p + 6
                V[v_idx][col] += coeff
            for loc_idx, coeff in schoolbook_W_local[r].items():
                if loc_idx == 0: w_idx = 6 * n + 0
                elif loc_idx == 1: w_idx = 6 * n + 1
                W[w_idx][col] += coeff
            col += 1
        return {"U": U, "V": V, "W": W}
