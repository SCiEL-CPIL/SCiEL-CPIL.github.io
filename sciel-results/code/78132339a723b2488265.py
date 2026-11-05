import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 88
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Function to add a standard Strassen decomposition for a 2x2x2 block
    # i_offset, j_offset, k_offset are the global starting indices
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices (0,0), (0,1), (1,0), (1,1)
        # Global U indices: (i_off + i_l) * m + (j_off + j_l)
        # Global V indices: (j_off + j_l) * p + (k_off + k_l)
        # Global W indices: (k_off + k_l) * n + (i_off + i_l)
        
        def u(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def v(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def w(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
        
        # M1: (A00+A11)(B00+B11)
        U[u(0,0), col] += 1; U[u(1,1), col] += 1
        V[v(0,0), col] += 1; V[v(1,1), col] += 1
        W[w(0,0), col] += 1; W[w(1,1), col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u(1,0), col] += 1; U[u(1,1), col] += 1
        V[v(0,0), col] += 1
        W[w(0,1), col] += 1; W[w(1,1), col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[u(0,0), col] += 1
        V[v(0,1), col] += 1; V[v(1,1), col] -= 1
        W[w(0,1), col] += 1; W[w(1,1), col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u(1,1), col] += 1
        V[v(1,0), col] += 1; V[v(0,0), col] -= 1
        W[w(0,0), col] += 1; W[w(0,1), col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u(0,0), col] += 1; U[u(0,1), col] += 1
        V[v(1,1), col] += 1
        W[w(0,0), col] -= 1; W[w(0,1), col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u(1,0), col] += 1; U[u(0,0), col] -= 1
        V[v(0,0), col] += 1; V[v(0,1), col] += 1
        W[w(1,1), col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u(0,1), col] += 1; U[u(1,1), col] -= 1
        V[v(1,0), col] += 1; V[v(1,1), col] += 1
        W[w(0,0), col] += 1
        col += 1

    # Part 1: Row 0 of A (i=0). 1x4x8.
    # This is just schoolbook: 32 terms.
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
    # Part 2: Rows 1,2 of A (i=1,2). 2x4x8.
    # Split into four 2x2x2 blocks? No, 2x4x2 blocks.
    # We split k into 4 blocks of size 2.
    # For each k-block (size 2), we have a 2x4x2 tensor.
    # We split the 2x4x2 into two 2x2x2 tensors (j=0,1 and j=2,3).
    # Each 2x2x2 uses 7 terms.
    # Total 4 blocks * 2 sub-blocks * 7 terms = 56 terms.
    
    for k_block in range(4):
        k_off = k_block * 2
        # Sub-block 1: j=0,1
        add_strassen_222(i_off=1, j_off=0, k_off=k_off)
        # Sub-block 2: j=2,3
        add_strassen_222(i_off=1, j_off=2, k_off=k_off)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
