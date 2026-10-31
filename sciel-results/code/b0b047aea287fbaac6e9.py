import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2 decomposition factors
    # U2, V2, W2 have shape (4, 7)
    # Rows are flattened indices of the 2x2 matrices
    # Cols are the 7 products
    
    # Initialize with zeros
    U2 = [[0] * 7 for _ in range(4)]
    V2 = [[0] * 7 for _ in range(4)]
    W2 = [[0] * 7 for _ in range(4)]
    
    # Term 1: P1 = (a+d)(e+h)
    # A: a(0,0), d(1,1) -> indices 0, 3
    # B: e(0,0), h(1,1) -> indices 0, 3
    # C: p(0,0), s(1,1) -> indices 0, 3
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: P2 = (c+d)e
    # A: c(1,0), d(1,1) -> indices 2, 3
    # B: e(0,0) -> index 0
    # C: r(1,0), s(1,1) -> indices 2, 3
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 3: P3 = a(f-h)
    # A: a(0,0) -> index 0
    # B: f(0,1), h(1,1) -> indices 1, 3
    # C: q(0,1), s(1,1) -> indices 1, 3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4: P4 = d(g-e)
    # A: d(1,1) -> index 3
    # B: g(1,0), e(0,0) -> indices 2, 0
    # C: p(0,0), r(1,0) -> indices 0, 2
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 5: P5 = (a+b)h
    # A: a(0,0), b(0,1) -> indices 0, 1
    # B: h(1,1) -> index 3
    # C: p(0,0), q(0,1) -> indices 0, 1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: P6 = (c-a)(e+f)
    # A: c(1,0), a(0,0) -> indices 2, 0
    # B: e(0,0), f(0,1) -> indices 0, 1
    # C: s(1,1) -> index 3
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: P7 = (b-d)(g+h)
    # A: b(0,1), d(1,1) -> indices 1, 3
    # B: g(1,0), h(1,1) -> indices 2, 3
    # C: p(0,0) -> index 0
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct U, V, W using Kronecker product of the 2x2 decomposition
    for r_b in range(7):
        for r_l in range(7):
            R = r_b * 7 + r_l
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    i_b = i // 2
                    i_l = i % 2
                    j_b = j // 2
                    j_l = j % 2
                    I_b = i_b * 2 + j_b
                    I_l = i_l * 2 + j_l
                    U[idx][R] = U2[I_b][r_b] * U2[I_l][r_l]
                    
            # Fill V
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    j_b = j // 2
                    j_l = j % 2
                    k_b = k // 2
                    k_l = k % 2
                    J_b = j_b * 2 + k_b
                    J_l = j_l * 2 + k_l
                    V[idx][R] = V2[J_b][r_b] * V2[J_l][r_l]
                    
            # Fill W
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    k_b = k // 2
                    k_l = k % 2
                    i_b = i // 2
                    i_l = i % 2
                    K_b = k_b * 2 + i_b
                    K_l = k_l * 2 + i_l
                    W[idx][R] = W2[K_b][r_b] * W2[K_l][r_l]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
