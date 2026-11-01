"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 18:
    # 1. Strassen for columns 0,1 (Rank 7)
    # 2. Strassen for columns 2,3 (Rank 7)
    # 3. Naive for column 4 (Rank 4)
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, rank_offset):
        # Strassen for 2x2x2 block starting at column col_offset
        # U indices: i*m + j
        # V indices: j*p + (col_offset + k)
        # W indices: (col_offset + k)*n + i
        
        # M1: (A11 + A22) * (B11 + B22)
        # A11 is U[0*m+0], A22 is U[1*m+1]
        # B11 is V[0*p + col_offset], B22 is V[1*p + col_offset + 1]
        # C11 uses M1, C22 uses M1
        
        U[0][rank_offset + 0] += 1
        U[3][rank_offset + 0] += 1
        V[col_offset][rank_offset + 0] += 1
        V[p + col_offset + 1][rank_offset + 0] += 1
        W[(col_offset)*n + 0][rank_offset + 0] += 1
        W[(col_offset + 1)*n + 1][rank_offset + 0] += 1

        # M2: (A11 + A22) * B21
        U[0][rank_offset + 1] += 1
        U[3][rank_offset + 1] += 1
        V[p + col_offset][rank_offset + 1] += 1
        W[(col_offset)*n + 1][rank_offset + 1] += 1
        W[(col_offset + 1)*n + 0][rank_offset + 1] -= 1 # C21 += M2, C11 -= M2

        # M3: (A11 + A21) * B11
        U[0][rank_offset + 2] += 1
        U[2][rank_offset + 2] += 1
        V[col_offset][rank_offset + 2] += 1
        W[(col_offset)*n + 0][rank_offset + 2] -= 1 # C11 -= M3
        W[(col_offset + 1)*n + 1][rank_offset + 2] -= 1 # C22 -= M3
        # Wait, Strassen reconstruction:
        # C11 = M1 - M2 + M4 - M6
        # C12 = M3 + M5 - M7
        # C21 = M2 + M6 - M4
        # C22 = M1 - M3 + M5 + M7
        
        # Let's re-verify Strassen formulas and signs.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A11+A22)B21
        # M3 = (A11+A21)B11
        # M4 = A22(B12+B21)
        # M5 = (A12+A22)B21
        # M6 = (A11-A21)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M2 + M4 - M6
        # C12 = M3 + M5 - M7
        # C21 = M2 + M6 - M4
        # C22 = M1 - M3 + M5 + M7
        
        # My W updates above for M1, M2, M3 were partial.
        # I will rewrite the helper to be cleaner.
        
    # Clear previous partial work if I had written it (I didn't, it was in head)
    # Let's implement cleanly.
    
    def add_strassen(col_start, r_start):
        # M1
        U[0][r_start + 0] += 1
        U[3][r_start + 0] += 1
        V[col_start][r_start + 0] += 1
        V[p + col_start + 1][r_start + 0] += 1
        W[col_start * n + 0][r_start + 0] += 1
        W[(col_start + 1) * n + 1][r_start + 0] += 1
        
        # M2
        U[0][r_start + 1] += 1
        U[3][r_start + 1] += 1
        V[p + col_start][r_start + 1] += 1
        W[col_start * n + 0][r_start + 1] -= 1
        W[(col_start + 1) * n + 1][r_start + 1] -= 1
        # Wait, C21 += M2. C21 is index (col_start, 1) in W?
        # W index is k*n + i.
        # C11 is i=0, k=col_start. W[col_start*n + 0].
        # C22 is i=1, k=col_start+1. W[(col_start+1)*n + 1].
        # C21 is i=1, k=col_start. W[col_start*n + 1].
        # C12 is i=0, k=col_start+1. W[(col_start+1)*n + 0].
        
        # Re-doing W mappings carefully.
        # C11 (i=0, k=0): W[0]
        # C12 (i=0, k=1): W[1]
        # C21 (i=1, k=0): W[2] (since n=2, 1*2+0=2)
        # C22 (i=1, k=1): W[3] (since n=2, 1*2+1=3)
        
        # M1 contributes to C11 (+1) and C22 (+1)
        W[col_start * n + 0][r_start + 0] += 1
        W[(col_start + 1) * n + 1][r_start + 0] += 1
        
        # M2 contributes to C11 (-1) and C21 (+1)
        W[col_start * n + 0][r_start + 1] -= 1
        W[col_start * n + 1][r_start + 1] += 1
        
        # M3 contributes to C12 (+1) and C22 (-1)
        W[(col_start + 1) * n + 0][r_start + 2] += 1
        W[(col_start + 1) * n + 1][r_start + 2] -= 1
        
        # M4: A22 * (B12 + B21)
        U[3][r_start + 3] += 1
        V[col_start + 1][r_start + 3] += 1
        V[p + col_start][r_start + 3] += 1
        W[col_start * n + 0][r_start + 3] += 1
        W[col_start * n + 1][r_start + 3] -= 1
        
        # M5: (A12 + A22) * B21
        U[1][r_start + 4] += 1
        U[3][r_start + 4] += 1
        V[p + col_start][r_start + 4] += 1
        W[(col_start + 1) * n + 0][r_start + 4] += 1
        W[(col_start + 1) * n + 1][r_start + 4] += 1
        
        # M6: (A11 - A21) * (B11 + B12)
        U[0][r_start + 5] += 1
        U[2][r_start + 5] -= 1
        V[col_start][r_start + 5] += 1
        V[col_start + 1][r_start + 5] += 1
        W[col_start * n + 0][r_start + 5] -= 1
        W[col_start * n + 1][r_start + 5] += 1
        
        # M7: (A12 - A22) * (B21 + B22)
        U[1][r_start + 6] += 1
        U[3][r_start + 6] -= 1
        V[p + col_start][r_start + 6] += 1
        V[p + col_start + 1][r_start + 6] += 1
        W[(col_start + 1) * n + 0][r_start + 6] -= 1
        W[(col_start + 1) * n + 1][r_start + 6] += 1

    # Add Strassen for cols 0,1 (rank 0-6)
    add_strassen(0, 0)
    
    # Add Strassen for cols 2,3 (rank 7-13)
    add_strassen(2, 7)
    
    # Add Naive for col 4 (rank 14-17)
    # C10 (i=0, k=4): W[4*n+0] = W[8]
    # C20 (i=1, k=4): W[4*n+1] = W[9]
    
    # N1: A11 * B14
    U[0][14] += 1
    V[4][14] += 1 # B[0,4] is row 0, col 4. V index 0*5+4=4.
    W[8][14] += 1
    
    # N2: A12 * B24
    U[1][15] += 1
    V[5+4][15] += 1 # B[1,4] is row 1, col 4. V index 1*5+4=9.
    W[8][15] += 1
    
    # N3: A21 * B14
    U[2][16] += 1
    V[4][16] += 1
    W[9][16] += 1
    
    # N4: A22 * B24
    U[3][17] += 1
    V[9][17] += 1
    W[9][17] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
