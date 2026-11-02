"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank 56 using Strassen decomposition for 8 independent 2x2 blocks
    rank = 56
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for a single 2x2 matrix product
    # M_s = sum_{i,j} U_s[i,j] A_{i,j} * sum_{j,r} V_s[j,r] B_{j,r}
    # C_{i,r} = sum_s W_s[i,r] M_s
    
    # U_s[i, j]
    U_coeffs = [
        [[1, 0], [0, 1]], # s=0: (A00+A11)
        [[0, 0], [1, 1]], # s=1: (A10+A11)
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: (A00+A01)
        [[0, 1], [-1, 0]],# s=5: (A01-A10)
        [[0, 1], [0, -1]] # s=6: (A10-A11)
    ]
    
    # V_s[j, r] (j is row of B, r is col of B within block)
    V_coeffs = [
        [[1, 0], [0, 1]], # s=0: (B00+B11)
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]],# s=2: (B01-B11)
        [[-1, 0], [1, 0]],# s=3: (B10-B00)
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: (B00+B01)
        [[0, 1], [1, 1]]  # s=6: (B11+B10)
    ]
    
    # W_s[i, r] (i is row of C, r is col of C within block)
    W_coeffs = [
        [[1, 0], [0, 1]], # s=0: C00+=M0, C11+=M0
        [[0, 0], [1, -1]],# s=1: C10+=M1, C11-=M1
        [[0, 1], [0, 1]], # s=2: C01+=M2, C11+=M2
        [[1, 0], [1, 0]], # s=3: C00+=M3, C10+=M3
        [[0, 1], [0, 1]], # s=4: C01+=M4, C11+=M4
        [[-1, 0], [0, 0]],# s=5: C00-=M5
        [[1, 0], [0, 0]]  # s=6: C00+=M6
    ]
    
    # Fill U, V, W
    for l in range(8): # 8 blocks
        for s in range(7): # 7 Strassen factors
            c = l * 7 + s
            
            # Fill U: index i*m + j = 2i + j
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][c] = U_coeffs[s][i][j]
                    
            # Fill V: index j*p + k. k = 2l + r
            for j in range(2):
                for r in range(2):
                    k = 2 * l + r
                    V[j * 16 + k][c] = V_coeffs[s][j][r]
                    
            # Fill W: index k*n + i. k = 2l + r
            for i in range(2):
                for r in range(2):
                    k = 2 * l + r
                    W[k * 2 + i][c] = W_coeffs[s][i][r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
