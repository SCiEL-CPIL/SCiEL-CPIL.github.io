"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 with rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    A is 2x2, B is 2x2, C = A*B.
    A indices: a00, a01, a10, a11 -> U index i*2+j
    B indices: b00, b01, b10, b11 -> V index j*2+k
    C indices: c00, c01, c10, c11 -> W index k*2+i
    """
    # Strassen:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b01 - b11)
    # M4 = a11 (b10 - b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[3, 0] = 1  # c11
    
    # M2 = (a10+a11) b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = -1 # c11
    
    # M3 = a00 (b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # M4 = a11 (b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    W[0, 3] = 1  # c00
    W[2, 3] = 1  # c10
    
    # M5 = (a00+a01) b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1 # c00
    W[1, 4] = 1  # c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1 # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # c00
    
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Build a decomposition for n x m * m x p using block decomposition.
    For 4x4 * 4x6: split into 3 blocks of 4x4 * 4x2, each using 4 Strassen 2x2*2x2.
    Total rank = 3 * 4 * 7 = 84.
    """
    # n=4, m=4, p=6
    # Split p into 3 groups of 2: p_groups = 3, p_per_group = 2
    # For each group, we have a 4x4 * 4x2 problem.
    # Split 4x4 into 4 blocks of 2x2 for A, and 4x2 into 2 blocks of 2x2 for B (but B is 4x2, so 2x2 blocks: 2 rows x 1 col of 2x2 blocks)
    # Actually: A is 4x4, B is 4x2.
    # A has 2x2 blocks: 2x2 = 4 blocks (A00, A01, A10, A11)
    # B has 2x2 blocks: 2x1 = 2 blocks (B00, B10) where B00 is top 2 rows, B10 is bottom 2 rows, each 2 cols wide.
    # C = A*B is 4x2. C has 2x2 blocks: 2x1 = 2 blocks (C00, C10).
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # Each product is 2x2 * 2x2, using Strassen rank 7.
    # So per group: 4 products * 7 = 28. Total: 3 * 28 = 84.
    
    n, m, p = 4, 4, 6
    p_groups = 3
    p_per_group = 2
    # For each group g (0,1,2), columns 2g, 2g+1 of B and C.
    # A is always the full 4x4.
    # B_g is 4x2 (columns 2g:2g+2 of B).
    # C_g is 4x2 (columns 2g:2g+2 of C).
    
    # Strassen 2x2*2x2
    sU, sV, sW = _strassen_2x2x2()
    
    total_rank = p_groups * 4 * 7
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # For each group g, for each of the 4 products (A_ij * B_kl), assign 7 columns.
    # A is 4x4: blocks A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B_g is 4x2: blocks B00 (rows 0-1, cols 0-1 of B_g), B10 (rows 2-3, cols 0-1 of B_g)
    # C_g is 4x2: blocks C00 (rows 0-1, cols 0-1 of C_g), C10 (rows 2-3, cols 0-1 of C_g)
    # Products:
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # So 4 products: (A00,B00), (A01,B10), (A10,B00), (A11,B10)
    
    product_defs = [
        # (a_row_block, a_col_block, b_row_block, c_row_block)
        (0, 0, 0, 0),  # A00 * B00 -> C00
        (0, 1, 1, 0),  # A01 * B10 -> C00
        (1, 0, 0, 1),  # A10 * B00 -> C10
        (1, 1, 1, 1),  # A11 * B10 -> C10
    ]
    
    col_offset = 0
    for g in range(p_groups):
        for (ar, ac, br, cr) in product_defs:
            # This product: A[ar*2:(ar+1)*2, ac*2:(ac+1)*2] * B_g[br*2:(br+1)*2, :] -> C_g[cr*2:(cr+1)*2, :]
            # A indices: row i = ar*2 + r, col j = ac*2 + s. U index = i*m + j = (ar*2+r)*4 + (ac*2+s)
            # B_g indices: row j = br*2 + s, col k = 0 or 1 (local). Global B col = g*2 + k. V index = j*p + k_global = (br*2+s)*6 + (g*2+k)
            # C_g indices: row i = cr*2 + r, col k = 0 or 1 (local). Global C col = g*2 + k. W index = k_global*n + i = (g*2+k)*4 + (cr*2+r)
            
            for c in range(7):
                # U: from sU[r*2+s, c] where r,s in {0,1}
                for r in range(2):
                    for s in range(2):
                        val = sU[r * 2 + s, c]
                        if val != 0:
                            i_global = ar * 2 + r
                            j_global = ac * 2 + s
                            U[i_global * m + j_global, col_offset + c] += val
                
                # V: from sV[s*2+k, c] where s in {0,1} (row of B), k in {0,1} (col of B)
                for s in range(2):
                    for k in range(2):
                        val = sV[s * 2 + k, c]
                        if val != 0:
                            j_global = br * 2 + s
                            k_global = g * 2 + k
                            V[j_global * p + k_global, col_offset + c] += val
                
                # W: from sW[k*2+r, c] where k in {0,1} (col of C), r in {0,1} (row of C)
                for k in range(2):
                    for r in range(2):
                        val = sW[k * 2 + r, c]
                        if val != 0:
                            i_global = cr * 2 + r
                            k_global = g * 2 + k
                            W[k_global * n + i_global, col_offset + c] += val
            
            col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140.0  # safety margin
    
    U, V, W = _build_block_decomposition(4, 4, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
