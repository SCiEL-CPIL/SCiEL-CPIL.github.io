import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct the decomposition using a block approach.
    # A is 2x4, B is 4x5.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into B_T (2x5) and B_B (2x5).
    # C = A_L B_T + A_R B_B.
    
    # For A_L B_T (2x2 * 2x5):
    # Split B_T into B_T1 (2x2), B_T2 (2x2), B_T3 (2x1).
    # Use Strassen (rank 7) for A_L B_T1 and A_L B_T2.
    # Use Schoolbook (rank 4) for A_L B_T3.
    # Total rank for this part: 7 + 7 + 4 = 18.
    
    # For A_R B_B (2x2 * 2x5):
    # Split B_B into B_B1 (2x2), B_B2 (2x2), B_B3 (2x1).
    # Use Strassen (rank 7) for A_R B_B1 and A_R B_B2.
    # Use Schoolbook (rank 4) for A_R B_B3.
    # Total rank for this part: 7 + 7 + 4 = 18.
    
    # Total Rank = 36.
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition
    # i_base: global row of A block (0 for 2x2 block)
    # j_base: global col of A block (0 or 2)
    # k_base: global col of B block (0 or 2)
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[i_base*4 + j_base][col] += 1
        U[i_base*4 + j_base+1][col] += 1
        V[j_base*5 + k_base][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        W[k_base*2 + i_base][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[i_base*4 + j_base+1][col] += 1
        U[i_base*4 + j_base+1][col] += 1
        V[j_base*5 + k_base][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[i_base*4 + j_base][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        V[j_base*5 + k_base+1][col] += -1
        W[k_base*2 + i_base][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[i_base*4 + j_base+1][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        V[j_base*5 + k_base][col] += -1
        W[k_base*2 + i_base][col] += -1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[i_base*4 + j_base][col] += 1
        U[i_base*4 + j_base+1][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        W[k_base*2 + i_base][col] += 1
        W[k_base*2 + i_base][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[i_base*4 + j_base+1][col] += 1
        U[i_base*4 + j_base][col] += -1
        V[j_base*5 + k_base][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i_base*4 + j_base+1][col] += 1
        U[i_base*4 + j_base+1][col] += -1
        V[j_base*5 + k_base+1][col] += 1
        V[j_base*5 + k_base+1][col] += 1
        W[k_base*2 + i_base][col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_221(i_base, j_base, k_base):
        nonlocal col
        # C_i,k = A_i0 B_0k + A_i1 B_1k
        # Term 1: A_i0 B_0k -> C_i,k
        # For i=0: A_00 B_0k -> C_0,k
        U[i_base*4 + j_base][col] += 1
        V[j_base*5 + k_base][col] += 1
        W[k_base*2 + i_base][col] += 1
        col += 1
        
        # Term 2: A_i1 B_1k -> C_i,k
        # For i=0: A_01 B_1k -> C_0,k
        U[i_base*4 + j_base+1][col] += 1
        V[(j_base+1)*5 + k_base][col] += 1
        W[k_base*2 + i_base][col] += 1
        col += 1
        
        # For i=1: A_10 B_0k -> C_1,k
        U[(i_base+1)*4 + j_base][col] += 1
        V[j_base*5 + k_base][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1
        
        # For i=1: A_11 B_1k -> C_1,k
        U[(i_base+1)*4 + j_base+1][col] += 1
        V[(j_base+1)*5 + k_base][col] += 1
        W[k_base*2 + i_base+1][col] += 1
        col += 1

    # Block 0: A_L (cols 0,1), B_T (rows 0,1)
    # Sub-block 0: k=0,1 (2x2x2)
    add_strassen(0, 0, 0)
    # Sub-block 1: k=2,3 (2x2x2)
    add_strassen(0, 0, 2)
    # Sub-block 2: k=4 (2x2x1)
    add_schoolbook_221(0, 0, 4)
    
    # Block 1: A_R (cols 2,3), B_B (rows 2,3)
    # Sub-block 0: k=0,1 (2x2x2)
    add_strassen(0, 2, 0)
    # Sub-block 1: k=2,3 (2x2x2)
    add_strassen(0, 2, 2)
    # Sub-block 2: k=4 (2x2x1)
    add_schoolbook_221(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
