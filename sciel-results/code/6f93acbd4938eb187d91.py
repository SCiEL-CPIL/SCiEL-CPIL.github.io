import time
_start_time = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition using the embedded Strassen approach
    if time.time() - _start_time < TIME_LIMIT - 20:
        try:
            import numpy as np
            
            # Rank 28 decomposition
            rank = 28
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            def add_term(u_dict, v_dict, w_dict, col):
                # u_dict: {(i, j): val}
                # v_dict: {(j, k): val}
                # w_dict: {(i, k): val}
                for (i, j), val in u_dict.items():
                    U[i * m + j][col] += val
                for (j, k), val in v_dict.items():
                    V[j * p + k][col] += val
                for (i, k), val in w_dict.items():
                    W[k * n + i][col] += val

            def get_strassen_terms(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
                terms = []
                
                # M1
                u = {(a_off_i+0, a_off_j+0): 1, (a_off_i+1, a_off_j+1): 1}
                v = {(b_off_j+0, b_off_k+0): 1, (b_off_j+1, b_off_k+1): 1}
                w = {(c_off_i+0, c_off_k+0): 1, (c_off_i+1, c_off_k+1): 1}
                terms.append((u, v, w))
                
                # M2
                u = {(a_off_i+1, a_off_j+0): 1, (a_off_i+1, a_off_j+1): 1}
                v = {(b_off_j+0, b_off_k+0): 1}
                w = {(c_off_i+1, c_off_k+0): 1}
                terms.append((u, v, w))
                
                # M3
                u = {(a_off_i+0, a_off_j+0): 1}
                v = {(b_off_j+0, b_off_k+1): 1, (b_off_j+1, b_off_k+1): -1}
                w = {(c_off_i+0, c_off_k+1): 1, (c_off_i+1, c_off_k+1): 1}
                terms.append((u, v, w))
                
                # M4
                u = {(a_off_i+1, a_off_j+1): 1}
                v = {(b_off_j+1, b_off_k+0): 1, (b_off_j+0, b_off_k+0): -1}
                w = {(c_off_i+0, c_off_k+0): 1, (c_off_i+1, c_off_k+0): 1}
                terms.append((u, v, w))
                
                # M5
                u = {(a_off_i+0, a_off_j+0): 1, (a_off_i+0, a_off_j+1): 1}
                v = {(b_off_j+1, b_off_k+1): 1}
                w = {(c_off_i+0, c_off_k+0): -1, (c_off_i+0, c_off_k+1): 1}
                terms.append((u, v, w))
                
                # M6
                u = {(a_off_i+1, a_off_j+0): 1, (a_off_i+0, a_off_j+0): -1}
                v = {(b_off_j+0, b_off_k+0): 1, (b_off_j+0, b_off_k+1): 1}
                w = {(c_off_i+1, c_off_k+1): 1}
                terms.append((u, v, w))
                
                # M7
                u = {(a_off_i+0, a_off_j+1): 1, (a_off_i+1, a_off_j+1): -1}
                v = {(b_off_j+1, b_off_k+0): 1, (b_off_j+1, b_off_k+1): 1}
                w = {(c_off_i+0, c_off_k+0): 1}
                terms.append((u, v, w))
                
                return terms

            col = 0
            # Block 1: A1 B1a
            for u, v, w in get_strassen_terms(0, 0, 0, 0, 0, 0):
                add_term(u, v, w, col)
                col += 1
                
            # Block 2: A1 B1b
            for u, v, w in get_strassen_terms(0, 0, 0, 2, 0, 2):
                add_term(u, v, w, col)
                col += 1
                
            # Block 3: A2 B2a
            for u, v, w in get_strassen_terms(0, 2, 2, 0, 0, 0):
                add_term(u, v, w, col)
                col += 1
                
            # Block 4: A2 B2b
            for u, v, w in get_strassen_terms(0, 2, 2, 2, 0, 2):
                add_term(u, v, w, col)
                col += 1
                
            if col == rank:
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return {"U": U_school, "V": V_school, "W": W_school}
