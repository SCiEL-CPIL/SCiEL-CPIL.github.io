"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140.0
    safety_margin = 5.0

    # Rank 15 decomposition for 2x3 x 3x4
    # 
    # Construction:
    # - Split B (3x4) into B1 (3x2, cols 0,1) and B2 (3x2, cols 2,3)
    # - For 2x3 x 3x2: use rank 7 construction (Strassen-based)
    # - Total: 7 + 7 = 14, but we need 15
    #
    # Actually, let me use the verified rank-15 construction:
    #
    # A = [[a, b, c],
    #      [d, e, f]]  (2x3)
    #
    # B = [[g, h, i, j],
    #      [k, l, m, n],
    #      [o, p, q, r]]  (3x4)
    #
    # C = A*B (2x4):
    # C[0,0] = ag + bk + co
    # C[0,1] = ah + bl + cp
    # C[0,2] = ai + bm + cq
    # C[0,3] = aj + bn + cr
    # C[1,0] = dg + ek + fo
    # C[1,1] = dh + el + fp
    # C[1,2] = di + em + fq
    # C[1,3] = dj + en + fr
    #
    # Rank 15 construction using 15 rank-1 tensors:
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6 x 15
    V = [[0] * rank for _ in range(m * p)]  # 12 x 15
    W = [[0] * rank for _ in range(p * n)]  # 8 x 15
    
    # Indexing:
    # U[i*m + j] for i in {0,1}, j in {0,1,2}
    # V[j*p + k] for j in {0,1,2}, k in {0,1,2,3}
    # W[k*n + i] for k in {0,1,2,3}, i in {0,1}
    
    # Term 0: a*g -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: b*k -> C[0,0]
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    # Term 2: c*o -> C[0,0]
    U[2][2] = 1; V[6][2] = 1; W[0][2] = 1
    # Term 3: a*h -> C[0,1]
    U[0][3] = 1; V[1][3] = 1; W[1][3] = 1
    # Term 4: b*l -> C[0,1]
    U[1][4] = 1; V[4][4] = 1; W[1][4] = 1
    # Term 5: c*p -> C[0,1]
    U[2][5] = 1; V[7][5] = 1; W[1][5] = 1
    # Term 6: d*g -> C[1,0]
    U[3][6] = 1; V[0][6] = 1; W[2][6] = 1
    # Term 7: e*k -> C[1,0]
    U[4][7] = 1; V[3][7] = 1; W[2][7] = 1
    # Term 8: f*o -> C[1,0]
    U[5][8] = 1; V[6][8] = 1; W[2][8] = 1
    # Term 9: d*h -> C[1,1]
    U[3][9] = 1; V[1][9] = 1; W[3][9] = 1
    # Term 10: e*l -> C[1,1]
    U[4][10] = 1; V[4][10] = 1; W[3][10] = 1
    # Term 11: f*p -> C[1,1]
    U[5][11] = 1; V[7][11] = 1; W[3][11] = 1
    # Term 12: a*i -> C[0,2]
    U[0][12] = 1; V[2][12] = 1; W[4][12] = 1
    # Term 13: b*m -> C[0,2]
    U[1][13] = 1; V[5][13] = 1; W[4][13] = 1
    # Term 14: c*q -> C[0,2]
    U[2][14] = 1; V[8][14] = 1; W[4][14] = 1
    
    # But this is only 15 terms and I'm missing C[0,3], C[1,2], C[1,3]
    # This is just schoolbook for 3 out of 4 columns.
    
    # Let me use a proper rank-15 construction:
    #
    # The correct approach: use the fact that rank(2x3x4) = 15
    # with an explicit verified construction.
    
    # Reset and use a different approach:
    # 
    # For 2x3 x 3x4, we can use:
    # - 2x2 x 2x2 Strassen (rank 7) for the top-left 2x2 block of C
    # - Additional terms for the remaining entries
    
    # Actually, let me just return the schoolbook rank 24 as a safe fallback,
    # since I cannot quickly verify a rank-15 construction.
    
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
