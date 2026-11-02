"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Level 2 (2x2 Strassen) Coefficients
    # u2[s][i][j] coeff of L[i,j] in left factor of M_s
    # v2[s][i][j] coeff of R[i,j] in right factor of M_s
    # w2[s][i][j] coeff of M_s in Z[i,j]
    
    u2 = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    v2 = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]
    
    w2 = [
        [[1, 0], [0, 1]],  # M1 -> Z00, Z11
        [[0, 0], [1, -1]], # M2 -> Z10, Z11
        [[0, 1], [1, 0]],  # M3 -> Z01, Z11
        [[1, 0], [1, 0]],  # M4 -> Z00, Z10
        [[-1, 1], [0, 0]], # M5 -> Z00, Z01
        [[0, 0], [0, 1]],  # M6 -> Z11
        [[1, 0], [0, 0]]   # M7 -> Z00
    ]
    
    # Level 1 (4x4 Blocks) Coefficients
    # l1[q][I][J] coeff of A[I,J] in L_q
    # v1[q][I][J] coeff of B[I,J] in R_q
    # w1[q][I][J] coeff of M_q in C[I,J]
    
    l1 = [
        [[1, 0], [0, 1]],  # L1 = A00 + A11
        [[0, 0], [1, 1]],  # L2 = A10 + A11
        [[1, 0], [0, 0]],  # L3 = A00
        [[0, 0], [0, 1]],  # L4 = A11
        [[1, 1], [0, 0]],  # L5 = A00 + A01
        [[-1, 0], [1, 0]], # L6 = A10 - A00
        [[0, 1], [0, -1]]  # L7 = A01 - A11
    ]
    
    v1 = [
        [[1, 0], [0, 1]],  # R1 = B00 + B11
        [[1, 0], [0, 0]],  # R2 = B00
        [[0, 1], [0, -1]], # R3 = B01 - B11
        [[-1, 0], [1, 0]], # R4 = B10 - B00
        [[0, 0], [0, 1]],  # R5 = B11
        [[1, 1], [0, 0]],  # R6 = B00 + B01
        [[0, 0], [1, 1]]   # R7 = B10 + B11
    ]
    
    w1 = [
        [[1, 0], [0, 1]],  # C00 += M1, C11 += M1
        [[0, 0], [1, -1]], # C10 += M2, C11 -= M2
        [[0, 1], [1, 0]],  # C01 += M3, C11 += M3
        [[1, 0], [1, 0]],  # C00 += M4, C10 += M4
        [[-1, 1], [0, 0]], # C00 -= M5, C01 += M5
        [[0, 0], [0, 1]],  # C11 += M6
        [[1, 0], [0, 0]]   # C00 += M7
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        q = r // 7
        s = r % 7
        
        for i in range(n):
            for j in range(m):
                I = i // 2
                J = j // 2
                ip = i % 2
                jp = j % 2
                U[i * m + j][r] = l1[q][I][J] * u2[s][ip][jp]
                
        for j in range(m):
            for k in range(p):
                I = j // 2
                K = k // 2
                jp = j % 2
                kp = k % 2
                V[j * p + k][r] = v1[q][I][K] * v2[s][jp][kp]
                
        for k in range(p):
            for i in range(n):
                K = k // 2
                I = i // 2
                kp = k % 2
                ip = i % 2
                W[k * n + i][r] = w1[q][I][K] * w2[s][ip][kp]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
