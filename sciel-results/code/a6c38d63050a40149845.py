import numpy as np

def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    
    # Define Strassen rank-7 decomposition
    # Indices: t (0..6), i (0..1), j (0..1) for U
    # t (0..6), j (0..1), k (0..1) for V
    # t (0..6), k (0..1), i (0..1) for W
    
    # U[t, i, j]
    U_s = np.zeros((7, 2, 2), dtype=int)
    # V[t, j, k]
    V_s = np.zeros((7, 2, 2), dtype=int)
    # W[t, k, i]
    W_s = np.zeros((7, 2, 2), dtype=int)
    
    # Term 0: M1 = (a+d)(e+h)
    U_s[0, 0, 0] = 1; U_s[0, 1, 1] = 1
    V_s[0, 0, 0] = 1; V_s[0, 1, 1] = 1
    W_s[0, 0, 0] = 1; W_s[0, 1, 1] = 1
    
    # Term 1: M2 = (c+d)e
    U_s[1, 1, 0] = 1; U_s[1, 1, 1] = 1
    V_s[1, 0, 0] = 1
    W_s[1, 0, 1] = 1; W_s[1, 1, 1] = -1
    
    # Term 2: M3 = a(f-h)
    U_s[2, 0, 0] = 1
    V_s[2, 0, 1] = 1; V_s[2, 1, 1] = -1
    W_s[2, 1, 0] = 1; W_s[2, 1, 1] = 1
    
    # Term 3: M4 = d(g-e)
    U_s[3, 1, 1] = 1
    V_s[3, 1, 0] = 1; V_s[3, 0, 0] = -1
    W_s[3, 0, 0] = 1; W_s[3, 0, 1] = 1
    
    # Term 4: M5 = (a+b)h
    U_s[4, 0, 0] = 1; U_s[4, 0, 1] = 1
    V_s[4, 1, 1] = 1
    W_s[4, 0, 0] = -1; W_s[4, 1, 0] = 1
    
    # Term 5: M6 = (c-a)(e+f)
    U_s[5, 1, 0] = 1; U_s[5, 0, 0] = -1
    V_s[5, 0, 0] = 1; V_s[5, 0, 1] = 1
    W_s[5, 1, 1] = 1
    
    # Term 6: M7 = (b-d)(g+h)
    U_s[6, 0, 1] = 1; U_s[6, 1, 1] = -1
    V_s[6, 1, 0] = 1; V_s[6, 1, 1] = 1
    W_s[6, 0, 0] = 1
    
    rank = num_blocks * 7
    
    # Initialize output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            # Fill U
            for i in range(2):
                for j in range(2):
                    if U_s[t, i, j] != 0:
                        U[i * m + j][col] = U_s[t, i, j]
            # Fill V
            for j in range(2):
                for k in range(2):
                    if V_s[t, j, k] != 0:
                        V[j * p + (2 * r + k)][col] = V_s[t, j, k]
            # Fill W
            for k in range(2):
                for i in range(2):
                    if W_s[t, k, i] != 0:
                        W[(2 * r + k) * n + i][col] = W_s[t, k, i]
                        
    return {"U": U, "V": V, "W": W}
