"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 4 independent 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen indices for 2x2 matrix A (rows 0,1; cols 0,1)
    # U index: i*m + j
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # For each of the 4 blocks of B (and C)
    for k in range(4):
        # B block k has columns 2k, 2k+1
        # V index: j*p + col
        # B00 (block k): 0*8 + 2k = 2k
        # B01 (block k): 0*8 + 2k+1 = 2k+1
        # B10 (block k): 1*8 + 2k = 8+2k
        # B11 (block k): 1*8 + 2k+1 = 8+2k+1
        
        # C block k has columns 2k, 2k+1
        # W index: col*n + row
        # C00 (block k, row 0, col 2k): 2k*2 + 0 = 4k
        # C01 (block k, row 0, col 2k+1): (2k+1)*2 + 0 = 4k+2
        # C10 (block k, row 1, col 2k): 2k*2 + 1 = 4k+1
        # C11 (block k, row 1, col 2k+1): (2k+1)*2 + 1 = 4k+3
        
        base_col = k * 7
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        c = base_col
        U[0][c] += 1
        U[3][c] += 1
        V[2*k][c] += 1
        V[8+2*k+1][c] += 1
        W[4*k][c] += 1
        W[4*k+3][c] += 1
        
        # M2: (A10+A11)B00 -> C10 += M2
        c = base_col + 1
        U[2][c] += 1
        U[3][c] += 1
        V[2*k][c] += 1
        W[4*k+1][c] += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        c = base_col + 2
        U[0][c] += 1
        V[2*k+1][c] += 1
        V[8+2*k+1][c] += -1
        W[4*k+2][c] += 1
        W[4*k+3][c] += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        c = base_col + 3
        U[3][c] += 1
        V[8+2*k][c] += 1
        V[2*k][c] += -1
        W[4*k][c] += 1
        W[4*k+1][c] += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        c = base_col + 4
        U[0][c] += 1
        U[1][c] += 1
        V[8+2*k+1][c] += 1
        W[4*k][c] += -1
        W[4*k+2][c] += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        c = base_col + 5
        U[2][c] += 1
        U[0][c] += -1
        V[2*k][c] += 1
        V[2*k+1][c] += 1
        W[4*k+3][c] += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        c = base_col + 6
        U[1][c] += 1
        U[3][c] += -1
        V[8+2*k][c] += 1
        V[8+2*k+1][c] += 1
        W[4*k][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
