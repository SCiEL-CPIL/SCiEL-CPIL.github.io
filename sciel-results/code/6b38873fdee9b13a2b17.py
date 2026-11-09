"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 6 blocks of 2x2x2 (Strassen) + 12 terms for 2x1x6
    # 6 * 7 + 12 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper for a 2x2x2 block
    # local i in {0,1}, local j in {0,1}, local k in {0,1}
    # Global offsets: i_off, j_off, k_off
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B10-B11) -> C01, C10
        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B01-B00) -> C00, C10
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A10)B11 -> -C00, C01
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + j_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6: (A11-A00)(B00+B10) -> C11
        U[(i_off+1) * m + (j_off+1)][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B11+B00) -> C00, C01
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1

    # Blocks for A1 (i=0,1; j=0,1) and A2 (i=0,1; j=2,3)
    # k splits into 0,1; 2,3; 4,5
    
    # A1, k=0,1
    add_strassen_block(0, 0, 0)
    # A1, k=2,3
    add_strassen_block(0, 0, 2)
    # A1, k=4,5
    add_strassen_block(0, 0, 4)
    
    # A2, k=0,1
    add_strassen_block(0, 2, 0)
    # A2, k=2,3
    add_strassen_block(0, 2, 2)
    # A2, k=4,5
    add_strassen_block(0, 2, 4)
    
    # A3 (i=0,1; j=4) and B3 (j=4; k=0..5)
    # C[i, k] += A[i, 4] * B[4, k]
    for k in range(6):
        # Term for i=0
        U[0 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term for i=1
        U[1 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
