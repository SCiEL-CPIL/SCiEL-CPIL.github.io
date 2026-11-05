"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 90 decomposition
    # Block 1: i in 0..2, j in 0..2, k in 0..6 (3x3x7)
    # Block 2: i in 0..2, j in 3..4, k in 0..6 (3x2x7)
    
    rank = 90
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Block 1: A[0:3, 0:3] * B[0:3, 0:7] ---
    # Partition k into 0..2, 3..5, 6
    # k=0..2: Schoolbook 3x3x3 (Rank 27)
    # k=3..5: Schoolbook 3x3x3 (Rank 27)
    # k=6: Schoolbook 3x3x1 (Rank 9)
    
    # k=0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # k=3..5
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # k=6
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    # --- Block 2: A[0:3, 3:5] * B[3:5, 0:7] ---
    # j indices in global A are 3, 4.
    # j indices in global B are 3, 4.
    # Partition A rows into 2 (rows 0,1) and 1 (row 2).
    # Part 1: A[0:2, 3:5] (2x2) * B[3:5, 0:7] (2x7)
    #   Use Strassen for k=0..1, 2..3, 4..5 (3 blocks of 2x2x2, rank 7 each = 21)
    #   Use Schoolbook for k=6 (2x2x1, rank 4)
    # Part 2: A[2:3, 3:5] (1x2) * B[3:5, 0:7] (2x7)
    #   Rank 2.
    
    # Part 1: 2x2x7
    # Strassen Block 1: k=0,1
    # Strassen coefficients for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    def add_strassen_block(i_base, j_base, k_start):
        nonlocal col
        # i_base: 0 or 1 (local row)
        # j_base: 3 or 4 (global col of A, row of B)
        # k_start: 0, 2, 4
        
        # M1
        U[(i_base) * m + j_base][col] = 1
        U[(i_base+1) * m + j_base][col] = 1
        V[j_base * p + k_start][col] = 1
        V[j_base * p + k_start+1][col] = 1
        V[(j_base+1) * p + k_start][col] = 1
        V[(j_base+1) * p + k_start+1][col] = 1
        W[k_start * n + i_base][col] = 1
        W[k_start * n + i_base+1][col] = 1
        W[(k_start+1) * n + i_base][col] = 1
        W[(k_start+1) * n + i_base+1][col] = 1
        col += 1
        
        # M2
        U[(i_base+1) * m + j_base][col] = 1
        U[(i_base+1) * m + j_base+1][col] = 1
        V[j_base * p + k_start][col] = 1
        W[k_start * n + i_base+1][col] = 1
        col += 1
        
        # M3
        U[i_base * m + j_base][col] = 1
        V[j_base * p + k_start+1][col] = 1
        V[(j_base+1) * p + k_start+1][col] = -1
        W[(k_start+1) * n + i_base][col] = 1
        col += 1
        
        # M4
        U[(i_base+1) * m + j_base+1][col] = 1
        V[(j_base+1) * p + k_start][col] = 1
        V[j_base * p + k_start][col] = -1
        W[k_start * n + i_base+1][col] = 1
        col += 1
        
        # M5
        U[i_base * m + j_base][col] = 1
        U[i_base * m + j_base+1][col] = 1
        V[(j_base+1) * p + k_start+1][col] = 1
        W[(k_start+1) * n + i_base][col] = 1
        col += 1
        
        # M6
        U[(i_base+1) * m + j_base][col] = 1
        U[i_base * m + j_base][col] = -1
        V[j_base * p + k_start][col] = 1
        V[j_base * p + k_start+1][col] = 1
        W[k_start * n + i_base+1][col] = 1
        W[(k_start+1) * n + i_base+1][col] = 1
        col += 1
        
        # M7
        U[i_base * m + j_base+1][col] = 1
        U[(i_base+1) * m + j_base+1][col] = -1
        V[(j_base+1) * p + k_start][col] = 1
        V[(j_base+1) * p + k_start+1][col] = 1
        W[(k_start+1) * n + i_base][col] = 1
        W[(k_start+1) * n + i_base+1][col] = 1
        col += 1

    # Strassen blocks for k=0,1; k=2,3; k=4,5
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Schoolbook for k=6 (2x2x1)
    for i_local in range(2):
        for j_local in range(2):
            U[(i_local) * m + (3 + j_local)][col] = 1
            V[(3 + j_local) * p + 6][col] = 1
            W[6 * n + i_local][col] = 1
            col += 1

    # Part 2: 1x2x7 (Row 2 of A block)
    # A[2, 3:5] * B[3:5, 0:7]
    # C[2, k] = A[2,3]B[3,k] + A[2,4]B[4,k]
    # Rank 2 terms:
    # Term 1: A[2,3] * B[3,k] -> W[2,k]
    U[2 * m + 3][col] = 1
    for k in range(7):
        V[3 * p + k][col] = 1
        W[k * n + 2][col] = 1
    col += 1
    
    # Term 2: A[2,4] * B[4,k] -> W[2,k]
    U[2 * m + 4][col] = 1
    for k in range(7):
        V[4 * p + k][col] = 1
        W[k * n + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
