import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 5 Strassen blocks (2x2x2) + 20 schoolbook terms (1x1x1)
    # Strassen rank 7, Schoolbook rank 1
    rank = 5 * 7 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    # j_base = 0
    # k_base = 2 * b
    for b in range(5):
        k_base = 2 * b
        # Local indices for A: (0,0), (0,1), (1,0), (1,1)
        # Global A indices:
        a_00 = 0 * m + 0
        a_01 = 0 * m + 1
        a_10 = 1 * m + 0
        a_11 = 1 * m + 1
        
        # Local indices for B: (0,0), (0,1), (1,0), (1,1)
        # Global B indices:
        b_00 = 0 * p + k_base
        b_01 = 0 * p + k_base + 1
        b_10 = 1 * p + k_base
        b_11 = 1 * p + k_base + 1
        
        # Local indices for C: (0,0), (0,1), (1,0), (1,1)
        # Global C indices:
        c_00 = k_base * n + 0
        c_01 = (k_base + 1) * n + 0
        c_10 = k_base * n + 1
        c_11 = (k_base + 1) * n + 1
        
        # Strassen terms
        # M1
        U[a_00][col] += 1
        U[a_11][col] += 1
        V[b_00][col] += 1
        V[b_11][col] += 1
        W[c_00][col] += 1
        W[c_11][col] += 1
        col += 1
        
        # M2
        U[a_10][col] += 1
        U[a_11][col] += 1
        V[b_00][col] += 1
        W[c_10][col] += 1
        W[c_11][col] -= 1
        col += 1
        
        # M3
        U[a_00][col] += 1
        V[b_01][col] += 1
        V[b_11][col] -= 1
        W[c_01][col] += 1
        W[c_11][col] += 1
        col += 1
        
        # M4
        U[a_11][col] += 1
        V[b_00][col] -= 1
        V[b_10][col] += 1
        W[c_00][col] += 1
        W[c_10][col] += 1
        col += 1
        
        # M5
        U[a_00][col] += 1
        U[a_01][col] += 1
        V[b_11][col] += 1
        W[c_00][col] -= 1
        W[c_01][col] += 1
        col += 1
        
        # M6
        U[a_00][col] -= 1
        U[a_10][col] += 1
        V[b_00][col] += 1
        V[b_01][col] += 1
        W[c_11][col] += 1
        col += 1
        
        # M7
        U[a_01][col] += 1
        U[a_11][col] -= 1
        V[b_10][col] += 1
        V[b_11][col] += 1
        W[c_00][col] += 1
        W[c_11][col] -= 1
        col += 1
        
    # Schoolbook terms for j=2
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
