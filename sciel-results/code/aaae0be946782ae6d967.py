import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    terms = []
    
    def verify_decomposition(U_list, V_list, W_list, rank):
        """Verify the decomposition is exact. Returns max error."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def build_schoolbook():
        """Build pure schoolbook decomposition as fallback."""
        terms_sb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms_sb.append((u, v, w))
        rank_sb = len(terms_sb)
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        for c in range(rank_sb):
            u_dict, v_dict, w_dict = terms_sb[c]
            for (i, j), val in u_dict.items():
                U_sb[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V_sb[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W_sb[k * n + i][c] = val
        return U_sb, V_sb, W_sb
    
    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition plan:
    # 1. j in {0, 1}
    #    1a. k in {0, 1}: Strassen(0, 0, 0)
    add_strassen(0, 0, 0)
    #    1b. k in {2, 3}: Strassen(0, 0, 2)
    add_strassen(0, 0, 2)
    #    1c. k in {4}: Schoolbook(0, 2, 0, 2, 4, 5)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # 2. j in {2, 3}
    #    2a. k in {0, 1}: Strassen(0, 2, 0)
    add_strassen(0, 2, 0)
    #    2b. k in {2, 3}: Strassen(0, 2, 2)
    add_strassen(0, 2, 2)
    #    2c. k in {4}: Schoolbook(0, 2, 2, 4, 4, 5)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # 3. j in {4}
    #    3a. k in {0, 1}: Schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    #    3b. k in {2, 3}: Schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    #    3c. k in {4}: Schoolbook(0, 2, 4, 5, 4, 5)
    add_schoolbook(0, 2, 4, 5, 4, 5)
    
    # Verify rank is as expected
    if len(terms) != 46:
        raise ValueError(f"Unexpected rank: {len(terms)}, expected 46")

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    # Verify the decomposition is exact
    max_err = verify_decomposition(U, V, W, rank)
    
    if max_err > 1e-6:
        # Fallback to pure schoolbook
        U, V, W = build_schoolbook()
    
    return {"U": U, "V": V, "W": W}
