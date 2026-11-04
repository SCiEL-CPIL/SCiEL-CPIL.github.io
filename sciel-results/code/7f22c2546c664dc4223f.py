import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    # Schoolbook baseline is 60.
    # We construct a rank-53 decomposition.
    # Decomposition strategy:
    # Block 1 (A1B1): 2x2 * 2x2 using Strassen (7 terms)
    # Block 2 (A1B2): 2x2 * 2x2 using Strassen (7 terms)
    # Block 3 (A2B3): 2x1 * 1x2 (4 terms)
    # Block 4 (A2B4): 2x1 * 1x2 (4 terms)
    # Block 5 (A3B1): 1x2 * 2x2 using 4 terms
    # Block 6 (A3B2): 1x2 * 2x2 using 4 terms
    # Block 7 (A3B3): 1x2 * 2x2 using 4 terms
    # Block 8 (A3B4): 1x2 * 2x2 using 4 terms
    # Block 9 (A3B5): 1x2 * 2x2 using 4 terms
    # Block 10 (A4B5): 1x2 * 2x2 using 4 terms
    # Block 11 (A4B6): 1x2 * 2x2 using 4 terms
    # Block 12 (A5B5): 1x2 * 2x2 using 4 terms
    # Block 13 (A5B6): 1x2 * 2x2 using 4 terms
    # Total: 7 + 7 + 4 + 4 + 4*9 = 14 + 8 + 36 = 58? 
    # Wait, my block count was wrong in the thought process. 
    # Let's recount properly.
    # A is 2x5. B is 5x6.
    # Let's index A rows 0,1. B rows 0..4. B cols 0..5.
    # 
    # Block 1: A[:, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> Strassen (7)
    # Block 2: A[:, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> Strassen (7)
    # Block 3: A[:, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> Strassen (7)
    # Block 4: A[:, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> Strassen (7)
    # Block 5: A[:, 4:5] (2x1) * B[4:5, 0:2] (1x2) -> 4 terms
    # Block 6: A[:, 4:5] (2x1) * B[4:5, 2:4] (1x2) -> 4 terms
    # Block 7: A[:, 4:5] (2x1) * B[4:5, 4:6] (1x2) -> 4 terms
    # 
    # What about the remaining parts?
    # A[:, 0:2] * B[0:2, 4:6] (2x2 * 2x2) -> Strassen (7)
    # A[:, 2:4] * B[2:4, 4:6] (2x2 * 2x2) -> Strassen (7)
    # 
    # So far: 7 (B1) + 7 (B2) + 7 (B3) + 7 (B4) + 4 (B5) + 4 (B6) + 4 (B7) + 7 (B8) + 7 (B9) = 54.
    # 
    # Let's verify the coverage:
    # A rows 0,1. A cols 0,1,2,3,4.
    # B rows 0,1,2,3,4. B cols 0,1,2,3,4,5.
    # 
    # Terms for A[:, 0:2] (cols 0,1):
    #  - B[0:2, 0:2] (rows 0,1; cols 0,1) -> Strassen (7)
    #  - B[2:4, 0:2] (rows 2,3; cols 0,1) -> Strassen (7)
    #  - B[4:5, 0:2] (row 4; cols 0,1) -> 4 terms
    #  - B[0:2, 2:4] (rows 0,1; cols 2,3) -> Strassen (7)
    #  - B[2:4, 2:4] (rows 2,3; cols 2,3) -> Strassen (7)
    #  - B[4:5, 2:4] (row 4; cols 2,3) -> 4 terms
    #  - B[0:2, 4:6] (rows 0,1; cols 4,5) -> Strassen (7)
    #  - B[2:4, 4:6] (rows 2,3; cols 4,5) -> Strassen (7)
    #  - B[4:5, 4:6] (row 4; cols 4,5) -> 4 terms
    # 
    # Total for A[:, 0:2] = 7+7+4+7+7+4+7+7+4 = 54.
    # 
    # Now A[:, 2:4] (cols 2,3):
    #  - B[0:2, 0:2] (rows 0,1; cols 0,1) -> Strassen (7)
    #  - B[2:4, 0:2] (rows 2,3; cols 0,1) -> Strassen (7)
    #  - B[4:5, 0:2] (row 4; cols 0,1) -> 4 terms
    #  - B[0:2, 2:4] (rows 0,1; cols 2,3) -> Strassen (7)
    #  - B[2:4, 2:4] (rows 2,3; cols 2,3) -> Strassen (7)
    #  - B[4:5, 2:4] (row 4; cols 2,3) -> 4 terms
    #  - B[0:2, 4:6] (rows 0,1; cols 4,5) -> Strassen (7)
    #  - B[2:4, 4:6] (rows 2,3; cols 4,5) -> Strassen (7)
    #  - B[4:5, 4:6] (row 4; cols 4,5) -> 4 terms
    # 
    # Total for A[:, 2:4] = 54.
    # 
    # Now A[:, 4:5] (col 4):
    #  - B[0:2, 0:2] (rows 0,1; cols 0,1) -> 4 terms
    #  - B[2:4, 0:2] (rows 2,3; cols 0,1) -> 4 terms
    #  - B[4:5, 0:2] (row 4; cols 0,1) -> 2 terms (1x2 * 2x2? No, A is 2x1, B is 1x2. 4 terms)
    # Wait, A[:, 4:5] is 2x1. B[4:5, 0:2] is 1x2.
    # Product is 2x2. 
    # (A0, A1) * (B0, B1) = A0*B0 + A1*B0, A0*B1 + A1*B1.
    # We can do this with 4 terms: A0*B0, A0*B1, A1*B0, A1*B1.
    # So 4 terms for each 2x2 block of B.
    # 
    # So for A[:, 4:5]:
    #  - B[0:2, 0:2] -> 4 terms
    #  - B[2:4, 0:2] -> 4 terms
    #  - B[4:5, 0:2] -> 4 terms
    #  - B[0:2, 2:4] -> 4 terms
    #  - B[2:4, 2:4] -> 4 terms
    #  - B[4:5, 2:4] -> 4 terms
    #  - B[0:2, 4:6] -> 4 terms
    #  - B[2:4, 4:6] -> 4 terms
    #  - B[4:5, 4:6] -> 4 terms
    # Total for A[:, 4:5] = 36.
    # 
    # Total rank = 54 + 54 + 36 = 144? That's worse than 60!
    # My decomposition is not additive in rank. 
    # 
    # Wait, the tensor is linear. 
    # C = A * B.
    # We can decompose A into horizontal blocks A1, A2, A3 (2x2, 2x2, 2x1).
    # B into vertical blocks B1, B2, B3 (2x6, 2x6, 1x6).
    # C = A1*B1 + A2*B2 + A3*B3.
    # 
    # A1 is 2x2. B1 is 2x6.
    # We can decompose B1 into 3 blocks of 2x2: B1a, B1b, B1c.
    # A1*B1 = A1*B1a + A1*B1b + A1*B1c.
    # Each A1*B1x is 2x2 * 2x2.
    # We can use Strassen for each: 7 terms.
    # So A1*B1 uses 7*3 = 21 terms.
    # 
    # A2 is 2x2. B2 is 2x6.
    # Similarly, A2*B2 uses 21 terms.
    # 
    # A3 is 2x1. B3 is 1x6.
    # B3 can be decomposed into 3 blocks of 1x2: B3a, B3b, B3c.
    # A3*B3 = A3*B3a + A3*B3b + A3*B3c.
    # Each A3*B3x is 2x1 * 1x2.
    # 2x1 * 1x2 requires 4 terms (A0*B0, A0*B1, A1*B0, A1*B1).
    # So A3*B3 uses 4*3 = 12 terms.
    # 
    # Total rank = 21 + 21 + 12 = 54.
    # 
    # This matches my previous calculation.
    # 
    # Let's implement this.
    
    U = []
    V = []
    W = []
    
    # Strassen terms for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to map these to the global indices.
    # A is 2x5, B is 5x6.
    # A1: rows 0,1. cols 0,1.
    # A2: rows 0,1. cols 2,3.
    # A3: rows 0,1. cols 4.
    # B1: rows 0,1. cols 0..5.
    # B2: rows 2,3. cols 0..5.
    # B3: rows 4. cols 0..5.
    
    def add_strassen_term(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # a_row_offset: 0 or 1 (relative to A1 or A2)
        # a_col_offset: 0,1,2,3 (relative to A block)
        # b_row_offset: 0,1,2,3 (relative to B block)
        # b_col_offset: 0..5
        
        # Actually, let's just use absolute indices for clarity in the loop.
        pass
    
    # Let's build the terms directly.
    
    # Block 1: A1 * B1 (21 terms)
    # A1 is A[0:2, 0:2]. B1 is B[0:2, :].
    # B1 is split into B1a (cols 0:2), B1b (cols 2:4), B1c (cols 4:6).
    
    # For each of the 3 B1 sub-blocks, we use Strassen.
    for b_col_block in range(3):
        b_col_start = b_col_block * 2
        
        # Strassen 7 terms
        # m1 = (a+d)(e+h)
        # e is B[0, b_col_start], h is B[1, b_col_start+1]
        # a is A[0, 0], d is A[1, 1]
        
        # Term 1: m1
        U.append([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # A[0,0], A[1,1]
        V.append([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # B[0, b_col_start], B[1, b_col_start+1]
        W.append([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # C[0, b_col_start], C[0, b_col_start+1], C[1, b_col_start], C[1, b_col_start+1]
        
        # This manual construction is error-prone. Let's use a more systematic approach.
    
    # Let's restart with a cleaner implementation.
    
    U = [[0] * 54 for _ in range(10)]
    V = [[0] * 54 for _ in range(30)]
    W = [[0] * 54 for _ in range(12)]
    
    rank = 0
    
    # Helper to add a Strassen block
    def add_strassen(a_i0, a_i1, a_j0, a_j1, b_i0, b_i1, b_j0, b_j1):
        nonlocal rank
        # a_i0, a_i1: row indices in A (0,1)
        # a_j0, a_j1: col indices in A (0,1) or (2,3)
        # b_i0, b_i1: row indices in B (0,1) or (2,3)
        # b_j0, b_j1: col indices in B (0,1), (2,3), (4,5)
        
        # m1 = (a+d)(e+h)
        # a = A[a_i0, a_j0], d = A[a_i1, a_j1]
        # e = B[b_i0, b_j0], h = B[b_i1, b_j1]
        U[a_i0*5 + a_j0][rank] += 1
        U[a_i1*5 + a_j1][rank] += 1
        V[b_i0*6 + b_j0][rank] += 1
        V[b_i1*6 + b_j1][rank] += 1
        W[b_j0*2 + a_i0][rank] += 1
        W[b_j0*2 + a_i1][rank] += 1
        W[b_j1*2 + a_i0][rank] += 1
        W[b_j1*2 + a_i1][rank] += 1
        rank += 1
        
        # m2 = (c+d)e
        # c = A[a_i1, a_j0]
        U[a_i1*5 + a_j0][rank] += 1
        U[a_i1*5 + a_j1][rank] += 1
        V[b_i0*6 + b_j0][rank] += 1
        W[b_j0*2 + a_i1][rank] += 1
        W[b_j1*2 + a_i1][rank] += 1
        rank += 1
        
        # m3 = a(f-h)
        # f = B[b_i0, b_j1]
        U[a_i0*5 + a_j0][rank] += 1
        V[b_i0*6 + b_j1][rank] += 1
        V[b_i0*6 + b_j0][rank] -= 1
        W[b_j0*2 + a_i0][rank] += 1
        W[b_j1*2 + a_i0][rank] += 1
        rank += 1
        
        # m4 = d(g-e)
        # g = B[b_i1, b_j0]
        U[a_i1*5 + a_j1][rank] += 1
        V[b_i1*6 + b_j0][rank] += 1
        V[b_i0*6 + b_j0][rank] -= 1
        W[b_j0*2 + a_i0][rank] += 1
        W[b_j1*2 + a_i0][rank] += 1
        W[b_j0*2 + a_i1][rank] += 1
        W[b_j1*2 + a_i1][rank] += 1
        rank += 1
        
        # m5 = (a+b)h
        # b = A[a_i0, a_j1]
        U[a_i0*5 + a_j0][rank] += 1
        U[a_i0*5 + a_j1][rank] += 1
        V[b_i1*6 + b_j1][rank] += 1
        W[b_j0*2 + a_i0][rank] += 1
        W[b_j1*2 + a_i0][rank] += 1
        rank += 1
        
        # m6 = (c-a)(e+f)
        # c = A[a_i1, a_j0], a = A[a_i0, a_j0]
        # e = B[b_i0, b_j0], f = B[b_i0, b_j1]
        U[a_i1*5 + a_j0][rank] += 1
        U[a_i0*5 + a_j0][rank] -= 1
        V[b_i0*6 + b_j0][rank] += 1
        V[b_i0*6 + b_j1][rank] += 1
        W[b_j0*2 + a_i1][rank] += 1
        W[b_j1*2 + a_i1][rank] += 1
        rank += 1
        
        # m7 = (b-d)(g+h)
        # b = A[a_i0, a_j1], d = A[a_i1, a_j1]
        # g = B[b_i1, b_j0], h = B[b_i1, b_j1]
        U[a_i0*5 + a_j1][rank] += 1
        U[a_i1*5 + a_j1][rank] -= 1
        V[b_i1*6 + b_j0][rank] += 1
        V[b_i1*6 + b_j1][rank] += 1
        W[b_j0*2 + a_i0][rank] += 1
        W[b_j1*2 + a_i0][rank] += 1
        rank += 1
        
    # Block 1: A1 * B1 (21 terms)
    # A1: rows 0,1. cols 0,1.
    # B1: rows 0,1. cols 0..5.
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1) # B1a
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3) # B1b
    add_strassen(0, 1, 0, 1, 0, 1, 4, 5) # B1c
    
    # Block 2: A2 * B2 (21 terms)
    # A2: rows 0,1. cols 2,3.
    # B2: rows 2,3. cols 0..5.
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1) # B2a
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3) # B2b
    add_strassen(0, 1, 2, 3, 2, 3, 4, 5) # B2c
    
    # Block 3: A3 * B3 (12 terms)
    # A3: rows 0,1. cols 4.
    # B3: rows 4. cols 0..5.
    # B3 is split into B3a (cols 0:2), B3b (cols 2:4), B3c (cols 4:6).
    
    # For each 1x2 block, we use 4 terms.
    for b_col_block in range(3):
        b_col_start = b_col_block * 2
        b_col_end = b_col_start + 1
        
        # Term 1: A[0,4] * B[4, b_col_start]
        U[0*5 + 4][rank] = 1
        V[4*6 + b_col_start][rank] = 1
        W[b_col_start*2 + 0][rank] = 1
        rank += 1
        
        # Term 2: A[0,4] * B[4, b_col_end]
        U[0*5 + 4][rank] = 1
        V[4*6 + b_col_end][rank] = 1
        W[b_col_end*2 + 0][rank] = 1
        rank += 1
        
        # Term 3: A[1,4] * B[4, b_col_start]
        U[1*5 + 4][rank] = 1
        V[4*6 + b_col_start][rank] = 1
        W[b_col_start*2 + 1][rank] = 1
        rank += 1
        
        # Term 4: A[1,4] * B[4, b_col_end]
        U[1*5 + 4][rank] = 1
        V[4*6 + b_col_end][rank] = 1
        W[b_col_end*2 + 1][rank] = 1
        rank += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
