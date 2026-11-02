import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Local indices i', j', k' in {0,1}
    # U row: i'*2 + j'
    # V row: j'*2 + k'
    # W row: k'*2 + i'
    
    # M1: (A00+A11)(B00+B11)
    # A00: i'=0, j'=0. A11: i'=1, j'=1
    # B00: j'=0, k'=0. B11: j'=1, k'=1
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal col
        # i_start is 0 or 1? No, i is always 0,1 for the 2x2 A block.
        # The block covers i in {0,1}.
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        U[i_start*4 + j_start][col] += 1
        U[i_start*4 + j_start + 1][col] += 1
        V[j_start*7 + k_start][col] += 1
        V[(j_start+1)*7 + (k_start+1)][col] += 1
        W[k_start*2 + i_start][col] += 1
        W[(k_start+1)*2 + (i_start+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        # A10: i'=1, j'=0. A11: i'=1, j'=1
        # B00: j'=0, k'=0
        U[i_start*4 + j_start][col] += 1 # Wait, A10 is i=1, j=0. i_start is 0?
        # Correction: The A block is 2x2, so it covers i=0,1.
        # So i_start is not a parameter for i, i is always 0,1 relative to the block?
        # No, the A matrix is 2x4.
        # A_1 is cols 0,1. A_2 is cols 2,3.
        # So for A_1, i is 0,1. For A_2, i is 0,1.
        # So i is always 0,1 in the global sense?
        # Yes, n=2.
        # So i_global = i_local.
        
        # Let's re-verify the mapping.
        # U row index: i_global * m + j_global.
        # i_global is 0 or 1.
        # j_global is j_start + j_local.
        
        # M2: (A10+A11)B00
        # A10: i=1, j=j_start. A11: i=1, j=j_start+1.
        # B00: j=j_start, k=k_start.
        U[1*4 + j_start][col] += 1
        U[1*4 + j_start + 1][col] += 1
        V[j_start*7 + k_start][col] += 1
        W[k_start*2 + 1][col] += 1
        col += 1
        
        # M3: A00(B01-B11)
        # A00: i=0, j=j_start.
        # B01: j=j_start, k=k_start+1. B11: j=j_start+1, k=k_start+1.
        U[0*4 + j_start][col] += 1
        V[j_start*7 + k_start + 1][col] += 1
        V[(j_start+1)*7 + k_start + 1][col] -= 1
        W[k_start + 1][col] += 1 # k=k_start+1, i=0 -> (k_start+1)*2 + 0
        col += 1
        
        # M4: A11(B10-B00)
        # A11: i=1, j=j_start+1.
        # B10: j=j_start+1, k=k_start. B00: j=j_start, k=k_start.
        U[1*4 + j_start + 1][col] += 1
        V[(j_start+1)*7 + k_start][col] += 1
        V[j_start*7 + k_start][col] -= 1
        W[k_start*2 + 1][col] += 1 # k=k_start, i=1
        col += 1
        
        # M5: (A00+A01)B11
        # A00: i=0, j=j_start. A01: i=0, j=j_start+1.
        # B11: j=j_start+1, k=k_start+1.
        U[0*4 + j_start][col] += 1
        U[0*4 + j_start + 1][col] += 1
        V[(j_start+1)*7 + k_start + 1][col] += 1
        W[(k_start+1)*2 + 0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # A10: i=1, j=j_start. A00: i=0, j=j_start.
        # B00: j=j_start, k=k_start. B01: j=j_start, k=k_start+1.
        U[1*4 + j_start][col] += 1
        U[0*4 + j_start][col] -= 1
        V[j_start*7 + k_start][col] += 1
        V[j_start*7 + k_start + 1][col] += 1
        W[k_start*2 + 1][col] += 1
        W[(k_start+1)*2 + 1][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # A01: i=0, j=j_start+1. A11: i=1, j=j_start+1.
        # B10: j=j_start+1, k=k_start. B11: j=j_start+1, k=k_start+1.
        U[0*4 + j_start + 1][col] += 1
        U[1*4 + j_start + 1][col] -= 1
        V[(j_start+1)*7 + k_start][col] += 1
        V[(j_start+1)*7 + k_start + 1][col] += 1
        W[k_start*2 + 0][col] += 1
        W[(k_start+1)*2 + 0][col] += 1
        col += 1

    # 2x2x1 block
    # i in {0,1}, j in {j_start, j_start+1}, k = k_single
    def add_2x2x1_block(j_start, k_single):
        nonlocal col
        # M1: A00 B00
        U[0*4 + j_start][col] += 1
        V[j_start*7 + k_single][col] += 1
        W[k_single*2 + 0][col] += 1
        col += 1
        
        # M2: A01 B01
        U[0*4 + j_start + 1][col] += 1
        V[(j_start+1)*7 + k_single][col] += 1
        W[k_single*2 + 0][col] += 1
        col += 1
        
        # M3: A10 B00
        U[1*4 + j_start][col] += 1
        V[j_start*7 + k_single][col] += 1
        W[k_single*2 + 1][col] += 1
        col += 1
        
        # M4: A11 B01
        U[1*4 + j_start + 1][col] += 1
        V[(j_start+1)*7 + k_single][col] += 1
        W[k_single*2 + 1][col] += 1
        col += 1

    # Apply blocks
    # A1 (j=0,1)
    add_strassen_block(0, 0, 0) # j=0,1; k=0,1
    add_strassen_block(0, 0, 2) # j=0,1; k=2,3
    add_strassen_block(0, 0, 4) # j=0,1; k=4,5
    add_2x2x1_block(0, 6)       # j=0,1; k=6
    
    # A2 (j=2,3)
    add_strassen_block(0, 2, 0) # j=2,3; k=0,1
    add_strassen_block(0, 2, 2) # j=2,3; k=2,3
    add_strassen_block(0, 2, 4) # j=2,3; k=4,5
    add_2x2x1_block(2, 6)       # j=2,3; k=6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
