import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    This uses the recursive Strassen algorithm: 7 outer multiplications of 2x2 blocks,
    each computed via 7 inner scalar multiplications.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to map 2x2 block indices to global 4x4 indices
    # A is n x m (4x4), B is m x p (4x4), C is p x n (4x4)
    # We view A as four 2x2 blocks: A00, A01, A10, A11
    # A[i, j] where i,j in {0,1} corresponds to A[2*i + di, 2*j + dj]
    
    def idx_a(i, j):
        return i * m + j
        
    def idx_b(i, j):
        return i * p + j
        
    def idx_c(i, j):
        return i * n + j
        
    # Strassen 2x2 multiplication formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose this into rank-7 terms for each 2x2 block multiplication.
    # The 2x2 Strassen decomposition itself has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # Let's define the 7 terms for a generic 2x2 multiplication X*Y = Z
    # X, Y, Z are 2x2 matrices.
    # The terms are linear combinations of entries of X and Y.
    
    # Term 1: (X00 + X11) * (Y00 + Y11) -> Z00, Z11
    # Term 2: (X10 + X11) * Y00 -> Z10, Z11
    # Term 3: X00 * (Y01 - Y11) -> Z01, Z11
    # Term 4: X11 * (Y10 - Y00) -> Z00, Z10
    # Term 5: (X00 + X01) * Y11 -> Z00, Z01
    # Term 6: (X10 - X00) * (Y00 + Y01) -> Z11
    # Term 7: (X01 - X11) * (Y10 + Y11) -> Z00
    
    # For each of the 7 outer terms, we apply the inner 7 terms.
    
    outer_terms = [
        # (X_coeffs, Y_coeffs, Z_contributions)
        # X_coeffs: dict mapping (i,j) to coeff
        # Y_coeffs: dict mapping (i,j) to coeff
        # Z_contributions: dict mapping (i,j) to coeff for this term's result added to Z
        {
            'X': {(0,0): 1, (1,1): 1},
            'Y': {(0,0): 1, (1,1): 1},
            'Z': {(0,0): 1, (1,1): 1} # M1 contributes to C00 and C11? No, M1 is the product.
            # Wait, the outer terms M1..M7 are the products.
            # C00 = M1 + M4 - M5 + M7
            # So M1 contributes +1 to C00.
            # M1 is a 2x2 matrix. Its entries contribute to the corresponding entries of the 2x2 block of C.
            # But C00 is a 2x2 block of the final 4x4 matrix C.
            # So M1 (a 2x2 matrix) adds directly to C00.
        }
    ]
    
    # Let's re-structure.
    # We have 7 outer products M_t = X_t * Y_t.
    # Each M_t is a 2x2 matrix.
    # The final C is composed of four 2x2 blocks C00, C01, C10, C11.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each outer term t, we know which block of C it goes to and with what sign.
    # Then, inside each outer term, we use Strassen's 7 multiplications.
    
    outer_definitions = [
        # t=0: M1
        {'X': {(0,0): 1, (1,1): 1}, 'Y': {(0,0): 1, (1,1): 1}, 'C_block': (0,0), 'C_sign': 1},
        {'X': {(0,0): 1, (1,1): 1}, 'Y': {(0,0): 1, (1,1): 1}, 'C_block': (1,1), 'C_sign': 1}, # M1 also in C11?
        # Wait, M1 is a single product. It contributes to C00 AND C11?
        # No, M1 is a 2x2 matrix.
        # C00 is a 2x2 matrix. C11 is a 2x2 matrix.
        # The formula C00 = M1 + ... means the entire 2x2 matrix M1 is added to C00.
        # And C11 = M1 - ... means the entire 2x2 matrix M1 is added to C11.
        # So M1 contributes to both C00 and C11.
        
        # Let's list the contributions of each M_t to the 4 blocks of C.
        # M1: C00 (+1), C11 (+1)
        # M2: C10 (+1), C11 (-1)
        # M3: C01 (+1), C11 (+1)
        # M4: C00 (+1), C10 (+1)
        # M5: C00 (-1), C01 (+1)
        # M6: C11 (+1)
        # M7: C00 (+1)
        
        # So:
        # M1 -> C00, C11
        # M2 -> C10, C11
        # M3 -> C01, C11
        # M4 -> C00, C10
        # M5 -> C00, C01
        # M6 -> C11
        # M7 -> C00
    ]
    
    # Correct mapping:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    contributions = {
        0: [(0,0, 1), (1,1, 1)],   # M1
        1: [(1,0, 1), (1,1, -1)],  # M2
        2: [(0,1, 1), (1,1, 1)],   # M3
        3: [(0,0, 1), (1,0, 1)],   # M4
        4: [(0,0, -1), (0,1, 1)],  # M5
        5: [(1,1, 1)],             # M6
        6: [(0,0, 1)],             # M7
    }
    
    # Inner Strassen terms for a 2x2 multiplication X*Y=Z
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    inner_contributions = {
        0: [(0,0, 1), (1,1, 1)],   # M1_inner
        1: [(1,0, 1), (1,1, -1)],  # M2_inner
        2: [(0,1, 1), (1,1, 1)],   # M3_inner
        3: [(0,0, 1), (1,0, 1)],   # M4_inner
        4: [(0,0, -1), (0,1, 1)],  # M5_inner
        5: [(1,1, 1)],             # M6_inner
        6: [(0,0, 1)],             # M7_inner
    }
    
    # Inner X and Y definitions
    # X1 = X00 + X11, Y1 = Y00 + Y11
    # X2 = X10 + X11, Y2 = Y00
    # X3 = X00, Y3 = Y01 - Y11
    # X4 = X11, Y4 = Y10 - Y00
    # X5 = X00 + X01, Y5 = Y11
    # X6 = X10 - X00, Y6 = Y00 + Y01
    # X7 = X01 - X11, Y7 = Y10 + Y11
    
    inner_X = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    
    inner_Y = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    
    # Outer X and Y definitions
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Note: A00 is the top-left 2x2 block of A.
    # In global indices, A00[i,j] is A[2*i, 2*j].
    # So X_coeffs for M1:
    # X1 uses A00 and A11.
    # A00 entries: (0,0), (0,1), (1,0), (1,1) in block coords -> (0,0), (0,1), (2,0), (2,1) in global?
    # No, block (0,0) covers rows 0,1 and cols 0,1.
    # Block (1,1) covers rows 2,3 and cols 2,3.
    
    outer_X = [
        # M1: A00 + A11
        {**{ (2*i, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) }},
        # M2: A10 + A11
        {**{ (2*i+2, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) }},
        # M3: A00
        { (2*i, 2*j): 1 for i in range(2) for j in range(2) },
        # M4: A11
        { (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) },
        # M5: A00 + A01
        {**{ (2*i, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i, 2*j+2): 1 for i in range(2) for j in range(2) }},
        # M6: A10 - A00
        {**{ (2*i+2, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i, 2*j): -1 for i in range(2) for j in range(2) }},
        # M7: A01 - A11
        {**{ (2*i, 2*j+2): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): -1 for i in range(2) for j in range(2) }},
    ]
    
    outer_Y = [
        # M1: B00 + B11
        {**{ (2*i, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) }},
        # M2: B00
        { (2*i, 2*j): 1 for i in range(2) for j in range(2) },
        # M3: B01 - B11
        {**{ (2*i, 2*j+2): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): -1 for i in range(2) for j in range(2) }},
        # M4: B10 - B00
        {**{ (2*i+2, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i, 2*j): -1 for i in range(2) for j in range(2) }},
        # M5: B11
        { (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) },
        # M6: B00 + B01
        {**{ (2*i, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i, 2*j+2): 1 for i in range(2) for j in range(2) }},
        # M7: B10 + B11
        {**{ (2*i+2, 2*j): 1 for i in range(2) for j in range(2) }, 
         **{ (2*i+2, 2*j+2): 1 for i in range(2) for j in range(2) }},
    ]
    
    col = 0
    for t in range(7):
        for k in range(7):
            # Combine outer and inner factors
            # U column col is the linear combination for the scalar multiplication
            # V column col is the linear combination for the scalar multiplication
            
            # U is for A (n x m = 4x4)
            # V is for B (m x p = 4x4)
            
            # The scalar multiplication is u_col . v_col
            # This scalar is added to specific entries of C.
            
            # The inner term k of outer term t corresponds to:
            # X_part = outer_X[t] combined with inner_X[k]
            # Y_part = outer_Y[t] combined with inner_Y[k]
            
            # Wait, the inner Strassen operates on the 2x2 blocks of the outer terms.
            # So the "X" in inner Strassen is the 2x2 matrix of linear combinations defined by outer_X[t].
            # No, outer_X[t] is a linear combination of the 4 blocks of A.
            # The result is a 2x2 matrix of linear combinations of A's entries.
            # Let's call this matrix X_t.
            # Then inner Strassen multiplies X_t * Y_t.
            # The inner terms are linear combinations of the entries of X_t and Y_t.
            
            # So, for a specific inner term k:
            # The factor for A is: sum over entries (i,j) in X_t of coeff_inner_X[k][i,j] * X_t[i,j]
            # Since X_t[i,j] is a linear combination of A's entries, we can substitute.
            
            # Let's pre-compute the global coefficients for U and V.
            
            # U_col[a_idx] = sum_{(bi, bj) in outer_X[t]} coeff_outer * sum_{(ii, ij) in inner_X[k] where (ii,ij) maps to (bi,bj)?}
            # No, it's simpler.
            
            # X_t is a 2x2 matrix where X_t[i,j] = sum_{a_idx} c_{t,i,j,a_idx} A[a_idx]
            # Inner term k uses X_k = sum_{i,j} d_{k,i,j} X_t[i,j]
            # So X_k = sum_{i,j} d_{k,i,j} sum_{a_idx} c_{t,i,j,a_idx} A[a_idx]
            # = sum_{a_idx} (sum_{i,j} d_{k,i,j} c_{t,i,j,a_idx}) A[a_idx]
            
            # So U_col[a_idx] = sum_{i,j} d_{k,i,j} c_{t,i,j,a_idx}
            
            # We need c_{t,i,j,a_idx}.
            # outer_X[t] is a dict {global_idx: coeff}.
            # But it's defined on the 4 blocks.
            # Block (bi, bj) corresponds to global indices (2*bi+ii, 2*bj+ij).
            # So c_{t, bi, bj, a_idx} is non-zero if a_idx is in block (bi, bj).
            # Specifically, if a_idx = (2*bi+ii, 2*bj+ij), then c_{t, bi, bj, a_idx} = outer_X[t][a_idx].
            
            # So U_col[a_idx] = outer_X[t].get(a_idx, 0) * inner_X[k].get(block_of(a_idx), 0)
            
            # Similarly for V.
            
            # And for W:
            # The product X_k * Y_k is a scalar.
            # This scalar is added to entries of Z_t (the 2x2 result of outer term t).
            # Z_t[li, lj] += inner_contributions[k][(li, lj)] * (X_k * Y_k)
            # Then C[global_i, global_j] += contributions[t][(block_i, block_j)] * Z_t[li, lj]
            # where global_i = 2*block_i + li, global_j = 2*block_j + lj.
            
            # So W_col[c_idx] = sum_{block_i, block_j, li, lj} 
            #      contributions[t][(block_i, block_j)] * inner_contributions[k][(li, lj)]
            #      if c_idx corresponds to (2*block_i+li, 2*block_j+lj)
            
            # Let's implement this.
            
            u_coeff = np.zeros(16)
            for a_idx, c_outer in outer_X[t].items():
                bi, bj = a_idx[0]//2, a_idx[1]//2
                c_inner = inner_X[k].get((bi, bj), 0)
                if c_inner != 0:
                    u_coeff[a_idx[0] * 4 + a_idx[1]] += c_outer * c_inner
            
            v_coeff = np.zeros(16)
            for b_idx, c_outer in outer_Y[t].items():
                bi, bj = b_idx[0]//2, b_idx[1]//2
                c_inner = inner_Y[k].get((bi, bj), 0)
                if c_inner != 0:
                    v_coeff[b_idx[0] * 4 + b_idx[1]] += c_outer * c_inner
            
            w_coeff = np.zeros(16)
            for (block_i, block_j, c_outer) in contributions[t]:
                for (li, lj, c_inner) in inner_contributions[k]:
                    gi = 2 * block_i + li
                    gj = 2 * block_j + lj
                    w_coeff[gi * 4 + gj] += c_outer * c_inner
            
            U[:, col] = u_coeff
            V[:, col] = v_coeff
            W[:, col] = w_coeff
            
            col += 1
            
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
