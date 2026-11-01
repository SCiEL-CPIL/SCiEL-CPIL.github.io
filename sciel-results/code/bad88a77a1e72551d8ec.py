import time
import random
import numpy as np

def _check_decomposition(U, V, W, n, m, p, num_tests=10):
    rank = len(U[0])
    for seed in range(num_tests):
        random.seed(seed * 7 + 13)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    u_val = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    v_val = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    s += W[k*n+i][r] * u_val * v_val
                C[i][k] = s
        
        if C != AB:
            return False
    return True

def _find_decomposition(r, n=2, m=2, p=6, max_attempts=10000):
    for attempt in range(max_attempts):
        U = np.random.randint(-2, 3, size=(n*m, r))
        V = np.random.randint(-2, 3, size=(m*p, r))
        
        W = np.zeros((p*n, r), dtype=float)
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((4, r))
                b_vec = np.zeros(4)
                
                for jj in range(m):
                    for jjp in range(m):
                        eq_idx = jj * m + jjp
                        for r_idx in range(r):
                            A_mat[eq_idx, r_idx] = U[i*m+jj, r_idx] * V[jj*p+k, r_idx]
                        b_vec[eq_idx] = 1.0 if jj == jjp else 0.0
                
                rank_sys = np.linalg.matrix_rank(A_mat)
                if rank_sys < 4:
                    valid = False
                    break
                
                W_col, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[k*n+i, :] = W_col
            
            if not valid:
                break
        
        if not valid:
            continue
        
        W_rounded = np.round(W * 2) / 2
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        
        U_list = [[float(x) for x in row] for row in U_rounded]
        V_list = [[float(x) for x in row] for row in V_rounded]
        W_list = [[float(x) for x in row] for row in W_rounded]
        
        if _check_decomposition(U_list, V_list, W_list, n, m, p):
            U_out = [[int(round(x*2))/2 for x in row] for row in U_rounded]
            V_out = [[int(round(x*2))/2 for x in row] for row in V_rounded]
            W_out = [[int(round(x*2))/2 for x in row] for row in W_rounded]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _strassen_2x2x6():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    v_s_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][v_s_idx]
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][w_s_idx]
                    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 130
    
    attempts_by_rank = {12: 50000, 13: 30000, 14: 20000, 15: 15000, 16: 10000, 17: 8000, 18: 5000, 19: 3000, 20: 2000}
    
    for r in range(12, 21):
        if time.time() > deadline - 5:
            break
        max_att = attempts_by_rank.get(r, 5000)
        result = _find_decomposition(r, n, m, p, max_attempts=max_att)
        if result is not None:
            return result
    
    result = _strassen_2x2x6()
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    return _schoolbook(n, m, p)
