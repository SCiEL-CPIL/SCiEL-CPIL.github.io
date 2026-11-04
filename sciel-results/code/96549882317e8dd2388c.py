import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(a_comb, b_comb, c_target):
        nonlocal col
        # a_comb: list of (global_a_idx, coeff)
        # b_comb: list of (global_b_idx, coeff)
        # c_target: (global_c_row, global_c_col_start, coeff)
        # The multiplication is (1x2) * (2x3)
        # a_comb defines the 1x2 matrix A_in.
        # b_comb defines the 2x3 matrix B_in.
        
        # We need to expand a_comb and b_comb into local variables a0, a1 and b00..b12
        # This is tricky because a_comb is a linear combination of global elements.
        # It's easier to just iterate the 6 schoolbook terms of the local multiplication.
        
        # Local A: [a0, a1]
        # Local B: [[b00, b01, b02], [b10, b11, b12]]
        # Local C: [c0, c1, c2]
        
        # The 6 products are:
        # a0*b00 -> c0
        # a0*b01 -> c1
        # a0*b02 -> c2
        # a1*b10 -> c0
        # a1*b11 -> c1
        # a1*b12 -> c2
        
        # We need to map a0, a1 to global A indices.
        # a0 is the first element of the 1x2 matrix.
        # a1 is the second element.
        
        # Similarly for b00, etc.
        
        # This requires knowing the structure of the combinations.
        # I will hardcode the 7 terms explicitly to avoid abstraction errors.
        
        pass

    # Term 0: (A11 + A22) * (B11 + B22)
    # A_in: a0 = A[0,0] + A[1,2], a1 = A[0,1] + A[1,3]
    # B_in: b00 = B[0,0] + B[2,3], b01 = B[0,1] + B[2,4], b02 = B[0,2] + B[2,5]
    #       b10 = B[1,0] + B[3,3], b11 = B[1,1] + B[3,4], b12 = B[1,2] + B[3,5]
    # C_out: c0 = C[0,0], c1 = C[0,1], c2 = C[0,2]
    
    # I will write a generic function that takes the local->global mappings.
    
    def process_term(a_map, b_map, c_map, coeff):
        nonlocal col
        # a_map: list of 2 lists. a_map[0] is list of (idx, coeff) for a0. a_map[1] for a1.
        # b_map: list of 2 lists of 3 lists. b_map[i][j] is list of (idx, coeff) for b_ij.
        # c_map: list of 3 lists. c_map[k] is list of (idx, coeff) for c_k.
        
        terms = [
            (0, 0, 0), # a0, b00, c0
            (0, 0, 1), # a0, b01, c1
            (0, 0, 2), # a0, b02, c2
            (1, 1, 0), # a1, b10, c0
            (1, 1, 1), # a1, b11, c1
            (1, 1, 2), # a1, b12, c2
        ]
        
        for ai, bi, ci in terms:
            # Add column
            for idx_a, coeff_a in a_map[ai]:
                U[idx_a][col] += coeff_a * coeff
            for idx_b, coeff_b in b_map[bi[0]][bi[1]]:
                V[idx_b][col] += coeff_b * coeff
            for idx_c, coeff_c in c_map[ci]:
                W[idx_c][col] += coeff_c * coeff
            col += 1

    # Define mappings
    # A11: (0,0), (0,1) -> idx 0, 1
    # A22: (1,2), (1,3) -> idx 6, 7
    # A12: (0,2), (0,3) -> idx 2, 3
    # A21: (1,0), (1,1) -> idx 4, 5
    
    # B11: (0,0)-(1,2) -> idx 0-5
    # B12: (0,3)-(1,5) -> idx 6-11
    # B21: (2,0)-(3,2) -> idx 12-17
    # B22: (2,3)-(3,5) -> idx 18-23
    
    # C11: (0,0)-(0,2) -> idx 0-2 (k*n+i: 0*2+0=0, 0*2+1=1, 0*2+2=2? No. W[k*n+i]. k is col, i is row.)
    # C is 2x6. W is 12x42.
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[0,2] -> k=2, i=0 -> idx 4
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[1,1] -> k=1, i=1 -> idx 3
    # C[1,2] -> k=2, i=1 -> idx 5
    # C[0,3] -> k=3, i=0 -> idx 6
    # ...
    
    # Term 0
    a0_map = [ (0, 1), (6, 1) ] # A[0,0] + A[1,2]
    a1_map = [ (1, 1), (7, 1) ] # A[0,1] + A[1,3]
    
    b00_map = [ (0, 1), (18, 1) ] # B[0,0] + B[2,3]
    b01_map = [ (1, 1), (19, 1) ]
    b02_map = [ (2, 1), (20, 1) ]
    b10_map = [ (3, 1), (21, 1) ]
    b11_map = [ (4, 1), (22, 1) ]
    b12_map = [ (5, 1), (23, 1) ]
    
    c0_map = [ (0, 1) ] # C[0,0]
    c1_map = [ (2, 1) ] # C[0,1]
    c2_map = [ (4, 1) ] # C[0,2]
    
    process_term([a0_map, a1_map], [[b00_map, b01_map, b02_map], [b10_map, b11_map, b12_map]], [c0_map, c1_map, c2_map], 1)
    
    # ... Repeat for other 6 terms ...
    
    return {"U": U, "V": V, "W": W}
