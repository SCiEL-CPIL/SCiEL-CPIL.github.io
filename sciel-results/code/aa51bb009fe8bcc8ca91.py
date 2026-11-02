import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard multiplication term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Helper to add a Strassen block for 2x2x2
    # local_i, local_j, local_k correspond to offsets in global i, j, k
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11 is (i_off, j_off), A22 is (i_off+1, j_off+1)
        # B11 is (j_off, k_off), B22 is (j_off+1, k_off+1)
        
        # We need to add 7 terms.
        # Each term is a sum of products.
        # This function adds the 7 rank-1 tensors corresponding to the 7 multiplications.
        
        # 1. (A11+A22) * (B11+B22)
        # This is not a single rank-1 term in the final U,V,W if we just write it down.
        # Wait, the Strassen decomposition defines 7 intermediate products M1..M7.
        # Each Mk is a product of a linear combo of A and a linear combo of B.
        # In the tensor decomposition, each Mk corresponds to ONE column in U, V, W.
        # The entries in that column are the coefficients of the linear combos.
        
        # M1: A_part = A11 + A22. B_part = B11 + B22.
        # U entries for A11, A22 are 1.
        # V entries for B11, B22 are 1.
        # W entries: C11 += M1, C22 += M1.
        
        # Let's list the 7 terms explicitly.
        
        # Term 1: M1
        # A: (i_off, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off) + (j_off+1, k_off+1)
        # C: (i_off, k_off) + (i_off+1, k_off+1)  <-- Wait, C11 and C22 get +M1.
        # But C12 and C21 get 0 from M1.
        
        # We fill U, V, W for this column.
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        
        W[k_off * n + i_off][col] = 1       # C11
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        
        col += 1

        # Term 2: M2 = (A21+A22) * B11
        # A: (i_off+1, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off)
        # C: C21 += M2, C22 -= M2
        
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        
        V[j_off * p + k_off][col] = 1
        
        W[k_off * n + (i_off+1)][col] = 1   # C21
        W[(k_off+1) * n + (i_off+1)][col] = -1 # C22
        
        col += 1

        # Term 3: M3 = A11 * (B12-B22)
        # A: (i_off, j_off)
        # B: (j_off, k_off+1) - (j_off+1, k_off+1)
        # C: C12 += M3, C22 += M3
        
        U[i_off * m + j_off][col] = 1
        
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        
        W[(k_off+1) * n + i_off][col] = 1   # C12
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        
        col += 1

        # Term 4: M4 = A22 * (B21-B11)
        # A: (i_off+1, j_off+1)
        # B: (j_off+1, k_off) - (j_off, k_off)
        # C: C11 += M4, C21 += M4
        
        U[(i_off+1) * m + (j_off+1)][col] = 1
        
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        
        W[k_off * n + i_off][col] = 1       # C11
        W[k_off * n + (i_off+1)][col] = 1   # C21
        
        col += 1

        # Term 5: M5 = (A11+A12) * B22
        # A: (i_off, j_off) + (i_off, j_off+1)
        # B: (j_off+1, k_off+1)
        # C: C11 -= M5, C12 += M5
        
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        
        V[(j_off+1) * p + (k_off+1)][col] = 1
        
        W[k_off * n + i_off][col] = -1     # C11
        W[(k_off+1) * n + i_off][col] = 1  # C12
        
        col += 1

        # Term 6: M6 = (A21-A11) * (B11+B12)
        # A: (i_off+1, j_off) - (i_off, j_off)
        # B: (j_off, k_off) + (j_off, k_off+1)
        # C: C22 += M6
        
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        
        col += 1

        # Term 7: M7 = (A12-A22) * (B21+B22)
        # A: (i_off, j_off+1) - (i_off+1, j_off+1)
        # B: (j_off+1, k_off) + (j_off+1, k_off+1)
        # C: C11 += M7
        
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        
        W[k_off * n + i_off][col] = 1       # C11
        
        col += 1

    # 1. Block C_bot: 1x3x4, Rank 12
    # i=2, j=0..2, k=0..3
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)

    # 2. Block C_top_R: 2x1x4, Rank 8
    # i=0..1, j=2, k=0..3
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)

    # 3. Block C_top_L_1: 2x2x2, Rank 7
    # i=0..1, j=0..1, k=0..1
    add_strassen(0, 0, 0)

    # 4. Block C_top_L_2: 2x2x2, Rank 7
    # i=0..1, j=0..1, k=2..3
    add_strassen(0, 0, 2)

    return {"U": U, "V": V, "W": W}
