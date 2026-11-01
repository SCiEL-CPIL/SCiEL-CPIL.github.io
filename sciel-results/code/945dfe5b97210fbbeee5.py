import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimation:
    # j=0 part: Rank 1
    # j=1,2 part (2x2x7):
    #   k=0,1 (2x2x2): Strassen Rank 7
    #   k=2 (2x2x1): Standard Rank 2
    #   k=3,4 (2x2x2): Strassen Rank 7
    #   k=5,6 (2x2x2): Strassen Rank 7
    # Total: 1 + 7 + 2 + 7 + 7 = 24
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0
    # U[i*3+0], V[0*7+k], W[k*2+i]
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: j=1,2
    # We need to handle 2x2x7 multiplication for A[:, 1:3] and B[1:3, :]
    # A_sub is 2x2, B_sub is 2x7
    # Split B_sub into B_01 (2x2), B_2 (2x1), B_34 (2x2), B_56 (2x2)
    
    # Helper to add standard 2x2x1 terms
    # A_sub[i, j_local], B_sub[j_local, k], C[i, k]
    # j_local 0 -> global j=1, j_local 1 -> global j=2
    
    def add_standard_2x2x1(k_offset):
        nonlocal col
        for i in range(2):
            for j_local in range(2):
                j_global = j_local + 1
                U[i * m + j_global][col] = 1
                V[j_global * p + (k_offset)][col] = 1
                W[(k_offset) * n + i][col] = 1
                col += 1

    # Helper to add Strassen 2x2x2 terms
    # A_sub[i, j_local], B_sub[j_local, k_local], C[i, k_local]
    # k_local 0,1 -> global k_offset, k_offset+1
    
    def add_strassen_2x2x2(k_offset):
        nonlocal col
        # Strassen identities for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A01 - A00)(B00 + B10)
        # M7 = (A10 - A11)(B01 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # We need to construct U, V, W for each M term.
        # U, V are linear combinations of A and B entries.
        # W has coefficients for C entries.
        
        # M1
        U[i * m + 1][col] = 1 # A00 (i=0, j=1)
        U[i * m + 2][col] = 0 # A01
        U[(i+1) * m + 1][col] = 1 # A10
        U[(i+1) * m + 2][col] = 0 # A11
        # Wait, Strassen sums A entries.
        # M1 uses (A00 + A11) and (B00 + B11).
        # This is not a rank-1 term in the tensor decomposition sense unless we expand?
        # No, Strassen works by defining new variables.
        # The tensor decomposition requires U, V, W to be constant vectors.
        # Strassen's 7 terms are:
        # 1. (A00+A11)(B00+B11) -> This is a product of sums.
        #    In tensor decomposition, this corresponds to a term where U has 1s at A00 and A11, V has 1s at B00 and B11.
        #    And W has coefficients for C00, C01, C10, C11.
        
        # Let's define the 7 Strassen terms explicitly.
        # Term 1: U: A00, A11. V: B00, B11. W: C00(+1), C01(+1), C10(+1), C11(-1)
        # Term 2: U: A10, A11. V: B00. W: C10(+1), C11(-1)
        # Term 3: U: A00. V: B10, B11. W: C01(+1), C10(-1)
        # Term 4: U: A11. V: B00, B01. W: C00(+1), C11(-1)
        # Term 5: U: A00, A10. V: B11. W: C00(-1), C10(+1)
        # Term 6: U: A01, A00. V: B00, B10. W: C11(+1)
        # Term 7: U: A10, A11. V: B01, B11. W: C00(+1)
        
        # Mapping indices:
        # A00 -> i=0, j=1 (global 1)
        # A01 -> i=0, j=2 (global 2)
        # A10 -> i=1, j=1 (global 4)
        # A11 -> i=1, j=2 (global 5)
        
        # B00 -> j=1, k=k_offset
        # B01 -> j=1, k=k_offset+1
        # B10 -> j=2, k=k_offset
        # B11 -> j=2, k=k_offset+1
        
        # C00 -> i=0, k=k_offset
        # C01 -> i=0, k=k_offset+1
        # C10 -> i=1, k=k_offset
        # C11 -> i=1, k=k_offset+1
        
        terms = [
            # (U_indices, V_indices, W_indices)
            # U: (0,1), (1,1) -> A00, A11
            ([(0, 1), (1, 2)], [(1, k_offset), (2, k_offset+1)], [(0, k_offset, 1), (0, k_offset+1, 1), (1, k_offset, 1), (1, k_offset+1, -1)]),
            # U: (1,1), (1,2) -> A10, A11
            ([(1, 1), (1, 2)], [(1, k_offset)], [(1, k_offset, 1), (1, k_offset+1, -1)]),
            # U: (0,1) -> A00
            ([(0, 1)], [(2, k_offset), (2, k_offset+1)], [(0, k_offset+1, 1), (1, k_offset, -1)]),
            # U: (1,2) -> A11
            ([(1, 2)], [(1, k_offset), (1, k_offset+1)], [(0, k_offset, 1), (1, k_offset+1, -1)]),
            # U: (0,1), (1,1) -> A00, A10
            ([(0, 1), (1, 1)], [(2, k_offset+1)], [(0, k_offset, -1), (1, k_offset, 1)]),
            # U: (0,2), (0,1) -> A01, A00
            ([(0, 2), (0, 1)], [(1, k_offset), (2, k_offset)], [(1, k_offset+1, 1)]),
            # U: (1,1), (1,2) -> A10, A11
            ([(1, 1), (1, 2)], [(1, k_offset+1), (2, k_offset+1)], [(0, k_offset, 1)])
        ]
        
        for u_inds, v_inds, w_inds in terms:
            for (i, j) in u_inds:
                U[i * m + j][col] = 1
            for (j, k) in v_inds:
                V[j * p + k][col] = 1
            for (i, k, val) in w_inds:
                W[k * n + i][col] = val
            col += 1

    # k=0,1
    add_strassen_2x2x2(0)
    # k=2
    add_standard_2x2x1(2)
    # k=3,4
    add_strassen_2x2x2(3)
    # k=5,6
    add_strassen_2x2x2(5)
    
    return {"U": U, "V": V, "W": W}
