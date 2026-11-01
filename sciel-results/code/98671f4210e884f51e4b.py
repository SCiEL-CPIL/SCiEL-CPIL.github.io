import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    target_rank = 7
    
    # Fallback Rank 12
    def get_fallback():
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Verification function
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = sum(U[u_idx][c] * V[v_idx][c] * W[w_idx][c] for c in range(rank))
                    if val != 1:
                        return False
        # Check zeros
        for row_u in range(n * m):
            for row_v in range(m * p):
                for row_w in range(p * n):
                    i = row_u // m
                    j = row_u % m
                    j_v = row_v // p
                    k = row_v % p
                    k_w = row_w // n
                    i_w = row_w % n
                    if j == j_v and i == i_w:
                        continue
                    val = sum(U[row_u][c] * V[row_v][c] * W[row_w][c] for c in range(rank))
                    if val != 0:
                        return False
        return True

    # Randomized Search
    # Try to find rank 7
    for _ in range(100): # 100 restarts
        U = np.random.randint(-2, 3, (n * m, target_rank))
        V = np.random.randint(-2, 3, (m * p, target_rank))
        W = np.random.randint(-2, 3, (p * n, target_rank))
        
        # Convert to lists for verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if verify(U_list, V_list, W_list, target_rank):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return get_fallback()
