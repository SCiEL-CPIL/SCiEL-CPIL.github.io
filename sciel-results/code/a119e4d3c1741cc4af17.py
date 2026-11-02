import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # We have 4 such blocks.
    # Block 1: A1 B1 -> C11 (cols 0-1)
    # Block 2: A2 B2 -> C11 (cols 0-1) [Summed with Block 1]
    # Block 3: A1 B3 -> C12 (cols 2-3)
    # Block 4: A2 B4 -> C12 (cols 2-3) [Summed with Block 3]
    
    # Helper to add a Strassen block
    def add_strassen_block(base_u, base_v, base_w, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        # a_row_offset, a_col_offset: offset in A (2x4)
        # b_row_offset, b_col_offset: offset in B (4x4)
        # c_col_offset: offset in C (2x4)
        
        # Indices in flattened U (A)
        # A is 2x4. Row i, Col j. Index i*4 + j.
        # A1 (2x2) at (0,0): indices 0,1,4,5
        # A2 (2x2) at (0,2): indices 2,3,6,7
        
        # Indices in flattened V (B)
        # B is 4x4. Row j, Col k. Index j*4 + k.
        
        # Indices in flattened W (C)
        # C is 2x4. Row i, Col k. Index k*2 + i.
        
        # Strassen terms
        # X = A_sub, Y = B_sub
        
        # Term 1: (X11+X22)(Y11+Y22)
        # U: X11, X22
        # V: Y11, Y22
        # W: C11, C12, C21, C22 (all 4)
        
        # We will fill U, V, W for 7 terms per block.
        
        # Term 1
        u_idx = [a_row_offset*4 + a_col_offset, (a_row_offset+1)*4 + (a_col_offset+1)]
        v_idx = [b_row_offset*4 + b_col_offset, (b_row_offset+1)*4 + (b_col_offset+1)]
        w_idx = [c_col_offset*2 + 0, c_col_offset*2 + 1, (c_col_offset+1)*2 + 0, (c_col_offset+1)*2 + 1]
        
        # Actually, it's easier to just iterate through the 7 Strassen formulas and assign coefficients.
        
        pass

    # Let's manually define the 28 terms.
    # This is tedious but robust.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1,4,5
    # B indices: 0,1,4,5
    # C indices: 0,1,2,3
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    # A indices: 2,3,6,7
    # B indices: 8,9,12,13
    # C indices: 0,1,2,3
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4]
    # A indices: 0,1,4,5
    # B indices: 2,3,6,7
    # C indices: 4,5,6,7
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    # A indices: 2,3,6,7
    # B indices: 10,11,14,15
    # C indices: 4,5,6,7

    # Strassen Coefficients
    # M1: (X11+X22)(Y11+Y22) -> C11, C12, C21, C22
    # M2: (X21+X22)Y11 -> C21, C22
    # M3: X11(Y12-Y22) -> C12, C22
    # M4: X22(Y21-Y11) -> C11, C21
    # M5: (X11+X12)Y22 -> C11, C12
    # M6: (X21-X11)(Y11+Y12) -> C22
    # M7: (X12-X22)(Y21+Y22) -> C11, C21

    def set_term(term_id, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (index, coeff)
        # v_coeffs: list of (index, coeff)
        # w_coeffs: list of (index, coeff)
        for idx, val in u_coeffs:
            U[idx, term_id] += val
        for idx, val in v_coeffs:
            V[idx, term_id] += val
        for idx, val in w_coeffs:
            W[idx, term_id] += val

    # Block 1 (Terms 0-6)
    # A: 0(11), 1(12), 4(21), 5(22)
    # B: 0(11), 1(12), 4(21), 5(22)
    # C: 0(11), 1(21), 2(12), 3(22)  <-- Note: W index is k*2+i.
    # C11 (i=0,k=0) -> 0
    # C12 (i=0,k=1) -> 2
    # C21 (i=1,k=0) -> 1
    # C22 (i=1,k=1) -> 3
    
    # T1: (A11+A22)(B11+B22)
    # U: 0(1), 5(1)
    # V: 0(1), 5(1)
    # W: 0(1), 2(1), 1(1), 3(1)
    set_term(0, [(0,1), (5,1)], [(0,1), (5,1)], [(0,1), (2,1), (1,1), (3,1)])
    
    # T2: (A21+A22)B11
    # U: 4(1), 5(1)
    # V: 0(1)
    # W: 1(1), 3(1)
    set_term(1, [(4,1), (5,1)], [(0,1)], [(1,1), (3,1)])
    
    # T3: A11(B12-B22)
    # U: 0(1)
    # V: 1(1), 5(-1)
    # W: 2(1), 3(1)
    set_term(2, [(0,1)], [(1,1), (5,-1)], [(2,1), (3,1)])
    
    # T4: A22(B21-B11)
    # U: 5(1)
    # V: 4(1), 0(-1)
    # W: 0(1), 1(1)
    set_term(3, [(5,1)], [(4,1), (0,-1)], [(0,1), (1,1)])
    
    # T5: (A11+A12)B22
    # U: 0(1), 1(1)
    # V: 5(1)
    # W: 0(-1), 2(1)  <-- Wait, C11 = M1 + M4 - M5 + M7. So M5 coeff in C11 is -1.
    # C12 = M3 + M5. So M5 coeff in C12 is 1.
    set_term(4, [(0,1), (1,1)], [(5,1)], [(0,-1), (2,1)])
    
    # T6: (A21-A11)(B11+B12)
    # U: 4(1), 0(-1)
    # V: 0(1), 1(1)
    # W: 3(1)  <-- C22 = M1 - M2 + M3 + M6. So M6 coeff is 1.
    set_term(5, [(4,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    
    # T7: (A12-A22)(B21+B22)
    # U: 1(1), 5(-1)
    # V: 4(1), 5(1)
    # W: 0(1), 1(1)  <-- C11 ... +M7. C21 ... +M7? No.
    # C11 = M1 + M4 - M5 + M7.
    # C21 = M2 + M4. Wait.
    # Let's re-verify Strassen formulas.
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    # So M7 only appears in Z11.
    # My previous set_term for T7 (index 6) should only touch W[0].
    set_term(6, [(1,1), (5,-1)], [(4,1), (5,1)], [(0,1)])

    # Block 2 (Terms 7-13)
    # A: 2(11), 3(12), 6(21), 7(22)
    # B: 8(11), 9(12), 12(21), 13(22)
    # C: 0,1,2,3 (Same as Block 1)
    
    set_term(7, [(2,1), (7,1)], [(8,1), (13,1)], [(0,1), (2,1), (1,1), (3,1)])
    set_term(8, [(6,1), (7,1)], [(8,1)], [(1,1), (3,1)])
    set_term(9, [(2,1)], [(9,1), (13,-1)], [(2,1), (3,1)])
    set_term(10, [(7,1)], [(12,1), (8,-1)], [(0,1), (1,1)])
    set_term(11, [(2,1), (3,1)], [(13,1)], [(0,-1), (2,1)])
    set_term(12, [(6,1), (2,-1)], [(8,1), (9,1)], [(3,1)])
    set_term(13, [(3,1), (7,-1)], [(12,1), (13,1)], [(0,1)])

    # Block 3 (Terms 14-20)
    # A: 0,1,4,5 (Same as Block 1)
    # B: 2(11), 3(12), 6(21), 7(22)
    # C: 4(11), 5(21), 6(12), 7(22)
    # W indices:
    # C11 (i=0,k=2) -> 2*2+0 = 4
    # C12 (i=0,k=3) -> 3*2+0 = 6
    # C21 (i=1,k=2) -> 2*2+1 = 5
    # C22 (i=1,k=3) -> 3*2+1 = 7
    
    set_term(14, [(0,1), (5,1)], [(2,1), (7,1)], [(4,1), (6,1), (5,1), (7,1)])
    set_term(15, [(4,1), (5,1)], [(2,1)], [(5,1), (7,1)])
    set_term(16, [(0,1)], [(3,1), (7,-1)], [(6,1), (7,1)])
    set_term(17, [(5,1)], [(6,1), (2,-1)], [(4,1), (5,1)])
    set_term(18, [(0,1), (1,1)], [(7,1)], [(4,-1), (6,1)])
    set_term(19, [(4,1), (0,-1)], [(2,1), (3,1)], [(7,1)])
    set_term(20, [(1,1), (5,-1)], [(6,1), (7,1)], [(4,1)])

    # Block 4 (Terms 21-27)
    # A: 2,3,6,7 (Same as Block 2)
    # B: 10(11), 11(12), 14(21), 15(22)
    # C: 4,5,6,7
    
    set_term(21, [(2,1), (7,1)], [(10,1), (15,1)], [(4,1), (6,1), (5,1), (7,1)])
    set_term(22, [(6,1), (7,1)], [(10,1)], [(5,1), (7,1)])
    set_term(23, [(2,1)], [(11,1), (15,-1)], [(6,1), (7,1)])
    set_term(24, [(7,1)], [(14,1), (10,-1)], [(4,1), (5,1)])
    set_term(25, [(2,1), (3,1)], [(15,1)], [(4,-1), (6,1)])
    set_term(26, [(6,1), (2,-1)], [(10,1), (11,1)], [(7,1)])
    set_term(27, [(3,1), (7,-1)], [(14,1), (15,1)], [(4,1)])

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
