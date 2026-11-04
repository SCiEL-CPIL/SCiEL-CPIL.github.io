"""Program for 4x5 * 5x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses block decomposition:
      A (4x5) = [A11 (4x4), A12 (4x1)]
      B (5x5) = [[B11 (4x4), B12 (4x1)],
                 [B21 (1x4), B22 (1x1)]]
      C (4x5) = [C11 (4x4), C12 (4x1)]
      
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    
    Uses Strassen (rank 7) for 2x2x2, recursively applied to get rank 49 for 4x4x4.
    Total rank: 49 (A11*B11) + 16 (A12*B21) + 16 (A11*B12) + 4 (A12*B22) = 85
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Each term is (u, v, w) where u, v, w are dicts
    # u: {(i,j): coeff} for A[i,j], v: {(j,k): coeff} for B[j,k], w: {(k,i): coeff} for C[i,k]
    # Note: w is indexed by (k,i) per the convention W[k*n+i, r]
    
    # Strassen 2x2 terms:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (-1)
    # M2 = (A10+A11)(B00)    -> contributes to C10 (+1), C11 (-1)
    # M3 = (A00)(B01-B11)    -> contributes to C01 (+1), C11 (+1)
    # M4 = (A11)(B10-B00)    -> contributes to C00 (+1), C10 (+1)
    # M5 = (A00+A01)(B11)    -> contributes to C00 (-1), C01 (+1)
    # M6 = (A10-A00)(B00+B01)-> contributes to C01 (+1), C11 (+1)
    # M7 = (A01-A11)(B10+B11)-> contributes to C00 (+1), C11 (+1)
    
    # Let me verify with explicit computation:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5 + M6
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6 + M7
    
    # Wait, let me re-derive carefully.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B01-B11) = A00B01 - A00B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 should be A00B00 + A01B10
    # C01 should be A00B01 + A01B11
    # C10 should be A10B00 + A11B10
    # C11 should be A10B01 + A11B11
    
    # C00 = M1 + M4 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10 ✓
    
    # C01 = M3 + M5 + M6
    # = (A00B01 - A00B11) + (A00B11 + A01B11) + (A10B00 + A10B01 - A00B00 - A00B01)
    # = A00B01 - A00B11 + A00B11 + A01B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A01B11 + A10B00 + A10B01 - A00B00
    # That's not right. C01 should be A00B01 + A01B11.
    
    # Let me try different combinations.
    # C01 = M3 + M5
    # = (A00B01 - A00B11) + (A00B11 + A01B11)
    # = A00B01 + A01B11 ✓
    
    # C10 = M2 + M4
    # = (A10B00 + A11B00) + (A11B10 - A11B00)
    # = A10B00 + A11B10 ✓
    
    # C11 = M1 - M2 + M3 + M6
    # = (A00B00 + A00B11 + A11B00 + A11B11) - (A10B00 + A11B00) + (A00B01 - A00B11) + (A10B00 + A10B01 - A00B00 - A00B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A11B11 + A10B01 ✓
    
    # So the correct assignments are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now express each M as a rank-1 term in the tensor:
    # M1: u1 = {(0,0):1, (1,1):1}, v1 = {(0,0):1, (1,1):1}, w1 = {(0,0):1, (1,1):-1}
    #   Wait, w is indexed by (k,i). For C00 (i=0,k=0): w[(0,0)], for C11 (i=1,k=1): w[(1,1)]
    #   M1 contributes +1 to C00 and -1 to C11.
    #   So w1 = {(0,0):1, (1,1):-1}
    
    # M2: u2 = {(1,0):1, (1,1):1}, v2 = {(0,0):1, (0,1):0, (1,0):0, (1,1):0} = {(0,0):1}
    #   Wait, M2 = (A10+A11)*B00. So u2 = {(1,0):1, (1,1):1}, v2 = {(0,0):1}
    #   M2 contributes +1 to C10 (i=1,k=0) and -1 to C11 (i=1,k=1)
    #   So w2 = {(0,1):1, (1,1):-1}  [w indexed by (k,i)]
    
    # M3: u3 = {(0,0):1}, v3 = {(0,1):1, (1,1):-1}
    #   M3 contributes +1 to C01 (i=0,k=1)
    #   So w3 = {(1,0):1}
    
    # M4: u4 = {(1,1):1}, v4 = {(1,0):1, (0,0):-1}
    #   M4 contributes +1 to C00 (i=0,k=0) and +1 to C10 (i=1,k=0)
    #   So w4 = {(0,0):1, (0,1):1}
    
    # M5: u5 = {(0,0):1, (0,1):1}, v5 = {(1,1):1}
    #   M5 contributes -1 to C00 (i=0,k=0) and +1 to C01 (i=0,k=1)
    #   So w5 = {(0,0):-1, (1,0):1}
    
    # M6: u6 = {(1,0):1, (0,0):-1}, v6 = {(0,0):1, (0,1):1}
    #   M6 contributes +1 to C11 (i=1,k=1)
    #   So w6 = {(1,1):1}
    
    # M7: u7 = {(0,1):1, (1,1):-1}, v7 = {(1,0):1, (1,1):1}
    #   M7 contributes +1 to C00 (i=0,k=0)
    #   So w7 = {(0,0):1}
    
    def strassen_2x2():
        """Return 7 rank-1 terms for 2x2x2 matrix multiplication."""
        terms = []
        # M1
        terms.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):-1}
        ))
        # M2
        terms.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1}
        ))
        # M3
        terms.append((
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1}
        ))
        # M4
        terms.append((
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1}
        ))
        # M5
        terms.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,0):-1, (1,0):1}
        ))
        # M6
        terms.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ))
        # M7
        terms.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1}
        ))
        return terms
    
    def expand_2x2_to_4x4(term_2x2, a_offset, b_offset, c_offset):
        """
        Expand a 2x2x2 term to a 4x4x4 term.
        a_offset: (row_offset, col_offset) for A in the 4x4 matrix
        b_offset: (row_offset, col_offset) for B in the 4x4 matrix  
        c_offset: (row_offset, col_offset) for C in the 4x4 matrix
        """
        u, v, w = term_2x2
        u_new = {}
        v_new = {}
        w_new = {}
        for (i, j), coeff in u.items():
            u_new[(i + a_offset[0], j + a_offset[1])] = coeff
        for (j, k), coeff in v.items():
            v_new[(j + b_offset[0], k + b_offset[1])] = coeff
        for (k, i), coeff in w.items():
            w_new[(k + c_offset[0], i + c_offset[1])] = coeff
        return (u_new, v_new, w_new)
    
    def strassen_4x4():
        """
        Generate 49 rank-1 terms for 4x4x4 matrix multiplication using recursive Strassen.
        Split 4x4 into 2x2 blocks, apply Strassen 7 products, each expanded with 2x2 Strassen.
        """
        s2 = strassen_2x2()
        terms = []
        
        # The 7 Strassen products for 4x4 (split into 2x2 blocks):
        # A = [[A11, A12], [A21, A22]] each 2x2
        # B = [[B11, B12], [B21, B22]] each 2x2
        # C = [[C11, C12], [C21, C22]] each 2x2
        #
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each of the 7 products, we need to expand the 2x2 linear forms
        # and then apply 2x2 Strassen to each resulting 2x2 product.
        
        # Product M1: (A11+A22)(B11+B22)
        # A factors: A11 (offset (0,0)) + A22 (offset (2,2))
        # B factors: B11 (offset (0,0)) + B22 (offset (2,2))
        # C output: C11 (offset (0,0)) + C22 (offset (2,2)) with signs +1 and -1
        
        # This is getting complex. Let me use a different approach.
        # I'll directly construct the 49 terms by enumerating all combinations.
        
        # Actually, let me just use schoolbook for 4x4x4 (64 terms) for simplicity,
        # and see if we can find a better construction later.
        # Total rank: 64 + 16 + 16 + 4 = 100. Same as seed.
        
        # Let me try to properly implement the recursive Strassen.
        # Each of the 7 products is a 2x2x2 multiplication, so 7*7=49 terms.
        
        # For product M1 = (A11+A22)(B11+B22):
        # The A part is a linear combination of A11 and A22.
        # The B part is a linear combination of B11 and B22.
        # When we expand (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Each of these is a 2x2x2 product.
        # But in Strassen, we don't expand - we treat (A11+A22) as one matrix and (B11+B22) as another.
        # So M1 is computed as a single 2x2x2 product of the summed matrices.
        
        # To generate the rank-1 terms for M1:
        # We need to express the 2x2 product of two matrices that are themselves linear combinations.
        # Let X = A11+A22, Y = B11+B22.
        # M1 = X*Y where X[i,j] = A11[i,j] + A22[i,j] and Y[i,j] = B11[i,j] + B22[i,j]
        # Using Strassen 2x2 on X*Y gives 7 terms.
        # Each term has u, v, w for the 2x2 product.
        # We need to "pull back" to the 4x4 indices.
        
        # For a term in the 2x2 product of X*Y:
        # u[(i,j)] contributes to X[i,j] = A11[i,j] + A22[i,j]
        #   So in the 4x4, u contributes to A11[i,j] (global (i,j)) and A22[i,j] (global (i+2,j+2))
        # v[(i,j)] contributes to Y[i,j] = B11[i,j] + B22[i,j]
        #   So in the 4x4, v contributes to B11[i,j] (global (i,j)) and B22[i,j] (global (i+2,j+2))
        # w[(k,i)] contributes to C11[k,i] (global (k,i)) and C22[k,i] (global (k+2,i+2))
        
        # Wait, but M1 contributes to C11 with +1 and C22 with -1.
        # So w should have +1 for C11 positions and -1 for C22 positions.
        
        # Let me define helper functions to expand terms.
        
        def expand_u_4x4(u_2x2, a_blocks):
            """
            Expand a 2x2 u term to 4x4.
            a_blocks: list of (offset, coeff) for A blocks
            """
            u_new = {}
            for (i, j), c in u_2x2.items():
                for (off, ac) in a_blocks:
                    u_new[(i + off[0], j + off[1])] = u_new.get((i + off[0], j + off[1]), 0) + c * ac
            return u_new
        
        def expand_v_4x4(v_2x2, b_blocks):
            """
            Expand a 2x2 v term to 4x4.
            b_blocks: list of (offset, coeff) for B blocks
            """
            v_new = {}
            for (i, j), c in v_2x2.items():
                for (off, bc) in b_blocks:
                    v_new[(i + off[0], j + off[1])] = v_new.get((i + off[0], j + off[1]), 0) + c * bc
            return v_new
        
        def expand_w_4x4(w_2x2, c_blocks):
            """
            Expand a 2x2 w term to 4x4.
            c_blocks: list of (offset, coeff) for C blocks
            """
            w_new = {}
            for (k, i), c in w_2x2.items():
                for (off, cc) in c_blocks:
                    w_new[(k + off[0], i + off[1])] = w_new.get((k + off[0], i + off[1]), 0) + c * cc
            return w_new
        
        # Now generate the 7 products:
        products = [
            # (a_blocks, b_blocks, c_blocks)
            # M1 = (A11+A22)(B11+B22)
            ([((0,0),1), ((2,2),1)], [((0,0),1), ((2,2),1)], [((0,0),1), ((2,2),-1)]),
            # M2 = (A21+A22)B11
            ([((2,0),1), ((2,2),1)], [((0,0),1)], [((2,0),1), ((2,2),-1)]),
            # M3 = A11(B12-B22)
            ([((0,0),1)], [((0,2),1), ((2,2),-1)], [((0,2),1), ((2,2),1)]),
            # M4 = A22(B21-B11)
            ([((2,2),1)], [((2,0),1), ((0,0),-1)], [((0,0),1), ((2,0),1)]),
            # M5 = (A11+A12)B22
            ([((0,0),1), ((0,2),1)], [((2,2),1)], [((0,0),-1), ((0,2),1)]),
            # M6 = (A21-A11)(B11+B12)
            ([((2,0),1), ((0,0),-1)], [((0,0),1), ((0,2),1)], [((2,2),1)]),
            # M7 = (A12-A22)(B21+B22)
            ([((0,2),1), ((2,2),-1)], [((2,0),1), ((2,2),1)], [((0,0),1)]),
        ]
        
        for a_blocks, b_blocks, c_blocks in products:
            for u2, v2, w2 in s2:
                u4 = expand_u_4x4(u2, a_blocks)
                v4 = expand_v_4x4(v2, b_blocks)
                w4 = expand_w_4x4(w2, c_blocks)
                terms.append((u4, v4, w4))
        
        return terms
    
    # Now build the full decomposition
    terms = []
    
    # Part 1: A11 * B11 (4x4 * 4x4 -> 4x4) using Strassen (49 terms)
    for u, v, w in strassen_4x4():
        terms.append((u, v, w))
    
    # Part 2: A12 * B21 (4x1 * 1x4 -> 4x4) - 16 terms
    for i in range(4):
        for k in range(4):
            u = {(i, 4): 1}
            v = {(4, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))
    
    # Part 3: A11 * B12 (4x4 * 4x1 -> 4x1) - 16 terms
    for i in range(4):
        for j in range(4):
            u = {(i, j): 1}
            v = {(j, 4): 1}
            w = {(4, i): 1}
            terms.append((u, v, w))
    
    # Part 4: A12 * B22 (4x1 * 1x1 -> 4x1) - 4 terms
    for i in range(4):
        u = {(i, 4): 1}
        v = {(4, 4): 1}
        w = {(4, i): 1}
        terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] += val
        for (j, k), val in v.items():
            V[j * p + k][r] += val
        for (k, i), val in w.items():
            W[k * n + i][r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
