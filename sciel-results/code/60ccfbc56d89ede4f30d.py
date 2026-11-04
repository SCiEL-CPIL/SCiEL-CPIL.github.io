"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # C1: 1x5x6 -> 30
    # C2: 2x5x6
    #   C2a: 2x2x6 -> 3 * Strassen(2x2x2) = 21
    #   C2b: 2x3x6
    #     C2b1: 2x1x6 -> 12
    #     C2b2: 2x2x6 -> 21
    # Total: 30 + 21 + 12 + 21 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, start_col):
        c = start_col
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[(i + i_off) * m + (j + j_off)][c] = 1
                    V[(j + j_off) * p + (k + k_off)][c] = 1
                    W[(k + k_off) * n + (i + i_off)][c] = 1
                    c += 1
        return c
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_off, j_off, k_off, start_col):
        # Strassen factors for 2x2x2
        # U indices: i*2 + j
        # V indices: j*2 + k
        # W indices: k*2 + i
        # 7 factors
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        c = start_col
        # Factor 1
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = 1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = -1
        c += 1
        # Factor 2
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1
        # Factor 3
        U[(i_off+0)*m + (j_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        c += 1
        # Factor 4
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = -1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        c += 1
        # Factor 5
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = -1
        c += 1
        # Factor 6
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+0)][c] = -1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = -1
        c += 1
        # Factor 7
        U[(i_off+0)*m + (j_off+1)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = -1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1
        return c

    # 1. C1: 1x5x6 (i=0, j=0..4, k=0..5)
    col = add_schoolbook(1, 5, 6, 0, 0, 0, col)
    
    # 2. C2a: 2x2x6 (i=1..2, j=0..1, k=0..5)
    # Split k into 2,2,2
    col = add_strassen(1, 0, 0, col)
    col = add_strassen(1, 0, 2, col)
    col = add_strassen(1, 0, 4, col)
    
    # 3. C2b1: 2x1x6 (i=1..2, j=2, k=0..5)
    col = add_schoolbook(2, 1, 6, 1, 2, 0, col)
    
    # 4. C2b2: 2x2x6 (i=1..2, j=3..4, k=0..5)
    # Split k into 2,2,2
    col = add_strassen(1, 3, 0, col)
    col = add_strassen(1, 3, 2, col)
    col = add_strassen(1, 3, 4, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
