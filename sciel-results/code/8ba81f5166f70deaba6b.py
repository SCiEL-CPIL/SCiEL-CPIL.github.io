import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We are decomposing A (4x5) * B (5x5)
    # A = [A1 | A2 | A3] where A1, A2 are 4x2, A3 is 4x1
    # B = [B11; B21; B31] where B11, B21 are 2x5, B31 is 1x5
    
    # Rank contributions:
    # A1*B11: 4x2 * 2x5. Split A1 into A11, A21 (2x2).
    # A11*B11 is 2x2 * 2x5. Split B11 into B111, B112 (2x2) and B113 (2x1).
    # A11*B111 (2x2x2, rank 7), A11*B112 (2x2x2, rank 7), A11*B113 (2x2x1, rank 4). Total 18.
    # A21*B11 similarly 18. Total 36.
    # A2*B21 similarly 36.
    # A3*B31: 4x1 * 1x5. Rank 4.
    # Total 76.
    
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(a_row_offset, b_row_offset, c_row_offset):
        # a_row_offset: start row of A in U (index in 4x5)
        # b_row_offset: start row of B in V (index in 5x5)
        # c_row_offset: start row of C in W (index in 5x4)
        # This adds 7 terms for 2x2x2 multiplication
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2
        # A indices: (a_row_offset, 0), (a_row_offset, 1), (a_row_offset+1, 0), (a_row_offset+1, 1) in local 2x2
        # But U is indexed by i*m+j.
        # For A11 (top-left 2x2 of A1 which is top-left 4x2 of A), A1 is cols 0,1 of A.
        # A11 is rows 0,1 of A1, so rows 0,1 of A, cols 0,1.
        # U index: i*5+j.
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map the 2x2 blocks to the global indices.
        
        # Let's define a helper to add a term
        def add_term(u_indices, v_indices, w_indices, signs):
            nonlocal col
            for u_idx, v_idx, w_idx, s in zip(u_indices, v_indices, w_indices, signs):
                U[u_idx][col] += s[0]
                V[v_idx][col] += s[1]
                W[w_idx][col] += s[2]
            col += 1
            
        # This is getting complex to generalize. I will hardcode the 76 terms.
        pass

    # I will write the explicit terms.
    # A is 4x5. B is 5x5.
    # A1: rows 0-3, cols 0-1.
    # A2: rows 0-3, cols 2-3.
    # A3: rows 0-3, col 4.
    
    # B11: rows 0-1, cols 0-4.
    # B21: rows 2-3, cols 0-4.
    # B31: row 4, cols 0-4.
    
    # Term 1: A1 * B11
    # A11 (rows 0-1, cols 0-1 of A) * B111 (rows 0-1, cols 0-1 of B) -> 2x2x2
    # A11 * B112 (rows 0-1, cols 2-3 of B) -> 2x2x2
    # A11 * B113 (rows 0-1, col 4 of B) -> 2x2x1
    # A21 (rows 2-3, cols 0-1 of A) * B111 -> 2x2x2
    # A21 * B112 -> 2x2x2
    # A21 * B113 -> 2x2x1
    
    # Helper for 2x2x1 (schoolbook)
    def add_2x2x1(a_r0, a_r1, b_r0, b_r1, c_r0, c_r1, c_col):
        # a_r0, a_r1 are row indices in A (0-3)
        # b_r0, b_r1 are row indices in B (0-4)
        # c_r0, c_r1 are row indices in C (0-4)
        # c_col is col index in C (0-4)
        nonlocal col
        # A is 2x1, B is 1x1? No, 2x2x1 means A is 2x1? No.
        # 2x2x1: A is 2x1? No, A is 2x1, B is 1x1?
        # Wait, 2x2x1 multiplication: A is 2x1, B is 1x1? No.
        # A is 2x1, B is 1x1? Product is 2x1.
        # My previous analysis: A11 is 2x2, B113 is 2x1.
        # A11 (2x2) * B113 (2x1) -> 2x1.
        # This is 4 multiplications.
        for i in range(2):
            for j in range(2):
                U[a_r0+i][col] = 1
                V[b_r0+j][col] = 1
                W[c_r0+i][col] = 1
                col += 1

    # Helper for 2x2x2 (Strassen)
    def add_2x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # a_r0, a_c0: top-left of A block (2x2)
        # b_r0, b_c0: top-left of B block (2x2)
        # c_r0, c_c0: top-left of C block (2x2)
        nonlocal col
        
        # M1
        U[a_r0][col] += 1; U[a_r0+1][col] += 1
        V[b_r0][col] += 1; V[b_r0+1][col] += 1
        W[c_r0][col] += 1; W[c_r0+1][col] += 1; W[c_r0][col] -= 1; W[c_r0+1][col] -= 1 # C11 += M1, C22 += M1
        # Wait, W indices are k*n+i.
        # C is 5x4.
        # C block is 2x2.
        # C11 is at (c_r0, c_c0), C12 at (c_r0, c_c0+1), C21 at (c_r0+1, c_c0), C22 at (c_r0+1, c_c0+1).
        # W index for C[k, i] is k*n+i.
        # So W[c_r0*n + c_c0], W[c_r0*n + c_c0+1], W[c_r0+1*n + c_c0], W[c_r0+1*n + c_c0+1].
        
        # Let's rewrite add_term properly.
        def term(u_list, v_list, w_list):
            nonlocal col
            for u, s in u_list: U[u][col] += s
            for v, s in v_list: V[v][col] += s
            for w, s in w_list: W[w][col] += s
            col += 1
            
        # M1 = (A00+A11)(B00+B11)
        term([(a_r0, 1), (a_r0+1, 1)], [(b_r0, 1), (b_r0+1, 1)], 
             [(c_r0*n+c_c0, 1), (c_r0+1*n+c_c0+1, 1)])
        # M2 = (A10+A11)B00
        term([(a_r0+1, 1), (a_r0+1+1, 1)], [(b_r0, 1)], 
             [(c_r0+1*n+c_c0, 1)])
        # M3 = A00(B10-B11)
        term([(a_r0, 1)], [(b_r0+1, 1), (b_r0+1+1, -1)], 
             [(c_r0*n+c_c0+1, 1), (c_r0+1*n+c_c0, 1)])
        # M4 = A11(B01-B00)
        term([(a_r0+1+1, 1)], [(b_r0+1, 1), (b_r0, -1)], 
             [(c_r0*n+c_c0, 1), (c_r0+1*n+c_c0+1, 1)])
        # M5 = (A00+A01)B11
        term([(a_r0, 1), (a_r0+1, 1)], [(b_r0+1+1, 1)], 
             [(c_r0*n+c_c0+1, 1), (c_r0+1*n+c_c0, -1)])
        # M6 = (A10-A00)(B00+B01)
        term([(a_r0+1, 1), (a_r0, -1)], [(b_r0, 1), (b_r0+1, 1)], 
             [(c_r0+1*n+c_c0+1, 1)])
        # M7 = (A01-A11)(B10+B11)
        term([(a_r0+1, 1), (a_r0+1+1, -1)], [(b_r0+1, 1), (b_r0+1+1, 1)], 
             [(c_r0*n+c_c0, -1), (c_r0+1*n+c_c0+1, 1)])
        
    # 1. A11 * B111 (2x2x2)
    add_2x2x2(0, 0, 0, 0, 0, 0)
    # 2. A11 * B112 (2x2x2)
    add_2x2x2(0, 0, 0, 2, 0, 2)
    # 3. A11 * B113 (2x2x1)
    add_2x2x1(0, 1, 0, 1, 0, 1, 4)
    # 4. A21 * B111 (2x2x2)
    add_2x2x2(2, 0, 0, 0, 2, 0)
    # 5. A21 * B112 (2x2x2)
    add_2x2x2(2, 0, 0, 2, 2, 2)
    # 6. A21 * B113 (2x2x1)
    add_2x2x1(2, 3, 0, 1, 2, 3, 4)
    
    # 7-12. A2 * B21 (same as 1-6 but shifted)
    add_2x2x2(0, 2, 2, 0, 0, 0)
    add_2x2x2(0, 2, 2, 2, 0, 2)
    add_2x2x1(0, 1, 2, 3, 0, 1, 4)
    add_2x2x2(2, 2, 2, 0, 2, 0)
    add_2x2x2(2, 2, 2, 2, 2, 2)
    add_2x2x1(2, 3, 2, 3, 2, 3, 4)
    
    # 13-16. A3 * B31 (4x1x1)
    for i in range(4):
        U[i][col] = 1
        V[4][col] = 1
        W[4*n+i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
