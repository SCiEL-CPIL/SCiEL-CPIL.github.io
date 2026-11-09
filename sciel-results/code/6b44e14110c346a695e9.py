"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def _add_strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    """
    Adds a rank-7 Strassen decomposition for a 2x2x2 sub-block.
    A rows: i0, i0+1
    A cols: j0, j0+1
    B rows: j0, j0+1
    B cols: k0, k0+1
    C rows: i0, i0+1
    C cols: k0, k0+1
    
    U index: i * 4 + j
    V index: j * 8 + k
    W index: k * 3 + i
    """
    # Local indices for A, B, C
    # A00: i0, j0 -> U[i0*4 + j0]
    # A01: i0, j0+1 -> U[i0*4 + j0 + 1]
    # A10: i0+1, j0 -> U[(i0+1)*4 + j0]
    # A11: i0+1, j0+1 -> U[(i0+1)*4 + j0 + 1]
    
    # B00: j0, k0 -> V[j0*8 + k0]
    # B01: j0, k0+1 -> V[j0*8 + k0 + 1]
    # B10: j0+1, k0 -> V[(j0+1)*8 + k0]
    # B11: j0+1, k0+1 -> V[(j0+1)*8 + k0 + 1]
    
    # C00: i0, k0 -> W[k0*3 + i0]
    # C01: i0, k0+1 -> W[(k0+1)*3 + i0]
    # C10: i0+1, k0 -> W[k0*3 + (i0+1)]
    # C11: i0+1, k0+1 -> W[(k0+1)*3 + (i0+1)]
    
    u00 = i0 * 4 + j0
    u01 = i0 * 4 + j0 + 1
    u10 = (i0 + 1) * 4 + j0
    u11 = (i0 + 1) * 4 + j0 + 1
    
    v00 = j0 * 8 + k0
    v01 = j0 * 8 + k0 + 1
    v10 = (j0 + 1) * 8 + k0
    v11 = (j0 + 1) * 8 + k0 + 1
    
    w00 = k0 * 3 + i0
    w01 = (k0 + 1) * 3 + i0
    w10 = k0 * 3 + (i0 + 1)
    w11 = (k0 + 1) * 3 + (i0 + 1)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[u00][offset] += 1.0
    U[u11][offset] += 1.0
    V[v00][offset] += 1.0
    V[v11][offset] += 1.0
    W[w00][offset] += 1.0
    W[w11][offset] += 1.0
    
    # M2 = (A10 + A11) * B00
    U[u10][offset + 1] += 1.0
    U[u11][offset + 1] += 1.0
    V[v00][offset + 1] += 1.0
    W[w10][offset + 1] += 1.0
    
    # M3 = A00 * (B01 - B11)
    U[u00][offset + 2] += 1.0
    V[v01][offset + 2] += 1.0
    V[v11][offset + 2] -= 1.0
    W[w01][offset + 2] += 1.0
    
    # M4 = A11 * (B10 - B00)
    U[u11][offset + 3] += 1.0
    V[v10][offset + 3] += 1.0
    V[v00][offset + 3] -= 1.0
    W[w00][offset + 3] += 1.0
    W[w10][offset + 3] += 1.0
    
    # M5 = (A00 + A01) * B11
    U[u00][offset + 4] += 1.0
    U[u01][offset + 4] += 1.0
    V[v11][offset + 4] += 1.0
    W[w00][offset + 4] -= 1.0
    W[w01][offset + 4] += 1.0
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[u10][offset + 5] += 1.0
    U[u00][offset + 5] -= 1.0
    V[v00][offset + 5] += 1.0
    V[v01][offset + 5] += 1.0
    W[w11][offset + 5] += 1.0
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[u01][offset + 6] += 1.0
    U[u11][offset + 6] -= 1.0
    V[v10][offset + 6] += 1.0
    V[v11][offset + 6] += 1.0
    W[w00][offset + 6] += 1.0
    W[w11][offset + 6] += 1.0

def _add_schoolbook_1x2x2(U, V, W, i0, j0, k0, offset):
    """
    Adds a rank-4 schoolbook decomposition for a 1x2x2 sub-block.
    A row: i0
    A cols: j0, j0+1
    B rows: j0, j0+1
    B cols: k0, k0+1
    C row: i0
    C cols: k0, k0+1
    """
    u0 = i0 * 4 + j0
    u1 = i0 * 4 + j0 + 1
    
    v00 = j0 * 8 + k0
    v01 = j0 * 8 + k0 + 1
    v10 = (j0 + 1) * 8 + k0
    v11 = (j0 + 1) * 8 + k0 + 1
    
    w0 = k0 * 3 + i0
    w1 = (k0 + 1) * 3 + i0
    
    # C0 = A0 B00 + A1 B10
    U[u0][offset] += 1.0
    V[v00][offset] += 1.0
    W[w0][offset] += 1.0
    
    U[u1][offset + 1] += 1.0
    V[v10][offset + 1] += 1.0
    W[w0][offset + 1] += 1.0
    
    # C1 = A0 B01 + A1 B11
    U[u0][offset + 2] += 1.0
    V[v01][offset + 2] += 1.0
    W[w1][offset + 2] += 1.0
    
    U[u1][offset + 3] += 1.0
    V[v11][offset + 3] += 1.0
    W[w1][offset + 3] += 1.0

def construct():
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    offset = 0
    
    # Loop over the two 4x4 blocks of B
    for b_block in range(2):
        k_base = b_block * 4
        
        # Top part of A (rows 0, 1)
        # Split A_top into A_t0 (cols 0,1) and A_t1 (cols 2,3)
        # Split B_block into B_00 (cols 0,1), B_01 (cols 2,3) relative to block
        # Note: B rows are 0..3.
        
        # C_top_left: i in 0..1, k in k_base..k_base+1
        # From A_t0 @ B_00 (j in 0..1, k in k_base..k_base+1)
        _add_strassen_2x2x2(U, V, W, 0, 0, k_base, offset)
        offset += 7
        
        # From A_t1 @ B_10 (j in 2..3, k in k_base..k_base+1)
        # Note: B rows for B_10 are 2,3. A cols for A_t1 are 2,3.
        _add_strassen_2x2x2(U, V, W, 0, 2, k_base, offset)
        offset += 7
        
        # C_top_right: i in 0..1, k in k_base+2..k_base+3
        # From A_t0 @ B_01 (j in 0..1, k in k_base+2..k_base+3)
        _add_strassen_2x2x2(U, V, W, 0, 0, k_base + 2, offset)
        offset += 7
        
        # From A_t1 @ B_11 (j in 2..3, k in k_base+2..k_base+3)
        _add_strassen_2x2x2(U, V, W, 0, 2, k_base + 2, offset)
        offset += 7
        
        # Bottom part of A (row 2)
        # Split A_bot into A_b0 (cols 0,1) and A_b1 (cols 2,3)
        
        # C_bot_left: i=2, k in k_base..k_base+1
        # From A_b0 @ B_00
        _add_schoolbook_1x2x2(U, V, W, 2, 0, k_base, offset)
        offset += 4
        
        # From A_b1 @ B_10
        _add_schoolbook_1x2x2(U, V, W, 2, 2, k_base, offset)
        offset += 4
        
        # C_bot_right: i=2, k in k_base+2..k_base+3
        # From A_b0 @ B_01
        _add_schoolbook_1x2x2(U, V, W, 2, 0, k_base + 2, offset)
        offset += 4
        
        # From A_b1 @ B_11
        _add_schoolbook_1x2x2(U, V, W, 2, 2, k_base + 2, offset)
        offset += 4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
