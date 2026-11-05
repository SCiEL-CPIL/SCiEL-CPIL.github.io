"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a 2x2 block decomposition with 3x3 blocks.
    # Strassen for 2x2 blocks (rank 7) tensor product Schoolbook for 3x3 blocks (rank 27).
    # Total rank = 7 * 27 = 189.
    
    rank_2x2 = 7
    rank_3x3 = 27
    total_rank = rank_2x2 * rank_3x3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen factors for 2x2 matrix multiplication.
    # Indices for 2x2: I, J in {0, 1}.
    # U2[I*2+J, c], V2[J*2+K, c], W2[K*2+I, c]
    # c is the Strassen step index 0..6.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U2 = [[0]*rank_2x2 for _ in range(4)]
    V2 = [[0]*rank_2x2 for _ in range(4)]
    W2 = [[0]*rank_2x2 for _ in range(4)]
    
    # Step 0 (M1)
    U2[0][0] = 1; U2[3][0] = 1 # A00, A11
    V2[0][0] = 1; V2[3][0] = 1 # B00, B11
    W2[0][0] = 1; W2[3][0] = 1 # C00, C11
    
    # Step 1 (M2)
    U2[2][1] = 1; U2[3][1] = 1 # A10, A11
    V2[0][1] = 1 # B00
    W2[2][1] = 1 # C10 (I=1, K=0 -> idx 2? No. W idx is K*2+I. K=0, I=1 -> 1. Wait.)
    # Let's re-verify W index mapping.
    # W[k*n+i, c]. For 2x2, n=2. W[k*2+i, c].
    # C10 is i=1, k=0. idx = 0*2 + 1 = 1.
    # C11 is i=1, k=1. idx = 1*2 + 1 = 3.
    # C00 is i=0, k=0. idx = 0.
    # C01 is i=0, k=1. idx = 1*2 + 0 = 2.
    
    # Correction on W indices for 2x2:
    # C00 -> 0
    # C01 -> 2
    # C10 -> 1
    # C11 -> 3
    
    # Re-do W2 assignments based on this.
    # M1 contributes to C00 (0) and C11 (3).
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 contributes to C10 (1).
    W2[1][1] = 1
    
    # M3 contributes to C01 (2).
    W2[2][2] = 1
    
    # M4 contributes to C00 (0) and C10 (1).
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5 contributes to C00 (0) with -1, and C01 (2) with +1.
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6 contributes to C11 (3).
    W2[3][5] = 1
    
    # M7 contributes to C00 (0) with +1, and C11 (3) with -1.
    W2[0][6] = 1; W2[3][6] = -1
    
    # Now fill U2 and V2 correctly.
    # M1: A00(0), A11(3). B00(0), B11(3).
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    
    # M2: A10(2), A11(3). B00(0).
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    
    # M3: A00(0). B01(1), B11(3) with -1.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    
    # M4: A11(3). B10(2), B00(0) with -1.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    
    # M5: A00(0), A01(1). B11(3).
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    
    # M6: A10(2) with 1, A00(0) with -1. B00(0), B01(1).
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    
    # M7: A01(1) with 1, A11(3) with -1. B10(2), B11(3).
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    
    # Define Schoolbook factors for 3x3 matrix multiplication.
    # Rank 27.
    # U3[u*3+v, d], V3[v*3+w, d], W3[w*3+u, d]
    # d corresponds to (u, v, w) triple.
    U3 = [[0]*rank_3x3 for _ in range(9)]
    V3 = [[0]*rank_3x3 for _ in range(9)]
    W3 = [[0]*rank_3x3 for _ in range(9)]
    
    col = 0
    for u in range(3):
        for v in range(3):
            for w in range(3):
                U3[u*3+v][col] = 1
                V3[v*3+w][col] = 1
                W3[w*3+u][col] = 1
                col += 1
                
    # Combine
    for c in range(rank_2x2):
        for d in range(rank_3x3):
            total_col = c * rank_3x3 + d
            
            for I in range(2):
                for J in range(2):
                    for u in range(3):
                        for v in range(3):
                            val = U2[I*2+J][c] * U3[u*3+v][d]
                            if val != 0:
                                i = 3*I + u
                                j = 3*J + v
                                U[i*m + j][total_col] = val
                                
            for J in range(2):
                for K in range(2):
                    for v in range(3):
                        for w in range(3):
                            val = V2[J*2+K][c] * V3[v*3+w][d]
                            if val != 0:
                                j = 3*J + v
                                k = 3*K + w
                                V[j*p + k][total_col] = val
                                
            for K in range(2):
                for I in range(2):
                    for w in range(3):
                        for u in range(3):
                            val = W2[K*2+I][c] * W3[w*3+u][d]
                            if val != 0:
                                k = 3*K + w
                                i = 3*I + u
                                W[k*n + i][total_col] = val
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
