import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # A (2x3) = [A1 (2x2), A2 (2x1)]
    # B (3x7) = [B1 (2x7); B2 (1x7)]
    # C = A1 B1 + A2 B2
    
    # A1 B1:
    # B1 (2x7) split into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)
    # Rank A1 B1a = 7 (Strassen)
    # Rank A1 B1b = 7 (Strassen)
    # Rank A1 B1c = 7 (Strassen)
    # Rank A1 B1d = 4 (Schoolbook)
    # Total 25
    
    # A2 B2:
    # A2 (2x1), B2 (1x7)
    # Rank 14 (Schoolbook)
    
    total_rank = 25 + 14
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for A1 (2x2) and B_block (2x2)
    # A1 indices: 0,1 (row 0), 2,3 (row 1) in A (flattened 2x3)
    # Actually A is 2x3.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A1 uses A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,3,4
    # A2 uses A[0,2], A[1,2] -> indices 2,5
    
    # B is 3x7.
    # B[0,0..6]=0..6
    # B[1,0..6]=7..13
    # B[2,0..6]=14..20
    # B1 uses B[0,:], B[1,:] -> indices 0..13
    # B2 uses B[2,:] -> indices 14..20
    
    def add_strassen(a_idx, b_idx_start, b_idx_end):
        # a_idx: list of 4 indices for A1 (r0c0, r0c1, r1c0, r1c1)
        # b_idx_start, b_idx_end: column range in B1 for the 2x2 block
        # B1 is rows 0,1 of B.
        # B indices for B1:
        # Row 0: 0..6
        # Row 1: 7..13
        # Block columns c1, c2
        # B_block[0,0] = B[0, c1]
        # B_block[0,1] = B[0, c2]
        # B_block[1,0] = B[1, c1]
        # B_block[1,1] = B[1, c2]
        
        nonlocal col
        # Strassen mults
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C = A1 B_block
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M7
        
        # A indices: a00, a01, a10, a11
        a00, a01, a10, a11 = a_idx
        
        # B indices: b00, b01, b10, b11
        c1 = b_idx_start
        c2 = b_idx_end
        b00 = c1 # B[0, c1]
        b01 = c2 # B[0, c2]
        b10 = 7 + c1 # B[1, c1]
        b11 = 7 + c2 # B[1, c2]
        
        mults = [
            # (coeffs for A, coeffs for B)
            ({a00: 1, a11: 1}, {b00: 1, b11: 1}), # M1
            ({a10: 1, a11: 1}, {b00: 1}),          # M2
            ({a00: 1}, {b01: 1, b11: -1}),         # M3
            ({a11: 1}, {b10: 1, b00: -1}),         # M4
            ({a00: 1, a01: 1}, {b11: 1}),          # M5
            ({a10: 1, a00: -1}, {b00: 1, b01: 1}), # M6
            ({a01: 1, a11: -1}, {b10: 1, b11: 1}), # M7
        ]
        
        # C indices for this block
        # C is 2x7.
        # Block corresponds to columns c1, c2 of C.
        # C[0, c1], C[0, c2], C[1, c1], C[1, c2]
        # W indices: k*n + i.
        # C[0, c1] -> k=c1, i=0 -> c1*2 + 0
        # C[0, c2] -> k=c2, i=0 -> c2*2 + 0
        # C[1, c1] -> k=c1, i=1 -> c1*2 + 1
        # C[1, c2] -> k=c2, i=1 -> c2*2 + 1
        
        w_c00 = c1 * 2 + 0
        w_c01 = c2 * 2 + 0
        w_c10 = c1 * 2 + 1
        w_c11 = c2 * 2 + 1
        
        w_coeffs = [
            {w_c00: 1, w_c11: -1, w_c10: -1, w_c01: 1}, # M1
            {w_c10: 1, w_c11: -1},                       # M2
            {w_c01: 1},                                  # M3
            {w_c00: -1, w_c10: 1},                       # M4
            {w_c00: -1, w_c01: 1},                       # M5
            {w_c11: 1},                                  # M6
            {w_c00: 1, w_c11: -1},                       # M7
        ]
        
        for i in range(7):
            a_c, b_c = mults[i]
            w_c = w_coeffs[i]
            
            for idx, val in a_c.items():
                U[idx][col] = val
            for idx, val in b_c.items():
                V[idx][col] = val
            for idx, val in w_c.items():
                W[idx][col] = val
            col += 1

    def add_schoolbook_2x2_2x1(a_idx, b_idx_col):
        # A1 (2x2) * B_col (2x1)
        # M1 = A00 B0
        # M2 = A01 B1
        # M3 = A10 B0
        # M4 = A11 B1
        # C00 = M1 + M2
        # C10 = M3 + M4
        
        nonlocal col
        a00, a01, a10, a11 = a_idx
        b0 = b_idx_col # B[0, col]
        b1 = 7 + b_idx_col # B[1, col]
        
        mults = [
            ({a00: 1}, {b0: 1}),
            ({a01: 1}, {b1: 1}),
            ({a10: 1}, {b0: 1}),
            ({a11: 1}, {b1: 1}),
        ]
        
        w_c00 = b_idx_col * 2 + 0
        w_c10 = b_idx_col * 2 + 1
        
        w_coeffs = [
            {w_c00: 1},
            {w_c00: 1},
            {w_c10: 1},
            {w_c10: 1},
        ]
        
        for i in range(4):
            a_c, b_c = mults[i]
            w_c = w_coeffs[i]
            
            for idx, val in a_c.items():
                U[idx][col] = val
            for idx, val in b_c.items():
                V[idx][col] = val
            for idx, val in w_c.items():
                W[idx][col] = val
            col += 1

    # A1 indices
    a1_idx = [0, 1, 3, 4]
    
    # B1 blocks
    # B1a: cols 0,1
    add_strassen(a1_idx, 0, 1)
    # B1b: cols 2,3
    add_strassen(a1_idx, 2, 3)
    # B1c: cols 4,5
    add_strassen(a1_idx, 4, 5)
    # B1d: col 6
    add_schoolbook_2x2_2x1(a1_idx, 6)
    
    # A2 B2
    # A2 indices: 2, 5
    # B2 indices: 14..20
    # C = A2 B2
    # C[i,k] = A[2+i, 2] * B[2, k]
    # A[0,2]=2, A[1,2]=5
    # B[2,k] = 14+k
    
    for k in range(7):
        b_idx = 14 + k
        # C[0,k] = A[0,2] B[2,k]
        U[2][col] = 1
        V[b_idx][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] = A[1,2] B[2,k]
        U[5][col] = 1
        V[b_idx][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
