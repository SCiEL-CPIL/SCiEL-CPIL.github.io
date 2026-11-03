import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition for 4x4 x 4x9.
    # Strategy: Split B into two parts: B1 (4x4) and B2 (4x5).
    # C = A B1 + A B2.
    # Decompose A B1 using Strassen (Rank 49).
    # Decompose A B2 using Schoolbook (Rank 80).
    # Total Rank = 129.
    
    rank1 = 49
    rank2 = 80
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # --- Part 1: 4x4x4 Strassen ---
    # We need to generate the 49 columns for the first 4 columns of B.
    # B is 4x4 in this part.
    # A is 4x4.
    # C is 4x4.
    
    # Strassen 4x4 formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product.
    # We decompose each 2x2 product using Strassen (Rank 7).
    
    # Strassen 2x2 formulas for (X, Y) -> Z:
    # s1 = (X00+X11)(Y00+Y11)
    # s2 = (X10+X11)Y00
    # s3 = X00(Y01-Y11)
    # s4 = X11(Y10-Y00)
    # s5 = (X00+X01)Y11
    # s6 = (X10-X00)(Y00+Y01)
    # s7 = (X01-X11)(Y10+Y11)
    
    # Z00 = s1 + s4 - s5 + s7
    # Z01 = s3 + s5
    # Z10 = s2 + s4
    # Z11 = s1 - s2 + s3 + s6
    
    # Indices for 4x4 matrices:
    # A[i, j], B[j, k], C[i, k] with i, j, k in 0..3.
    # Blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # C00: i in 0,1; k in 0,1
    # C01: i in 0,1; k in 2,3
    # C10: i in 2,3; k in 0,1
    # C11: i in 2,3; k in 2,3
    
    # Outer terms definitions:
    # List of (coeff, block_idx) for A and B.
    # block_idx: 0->00, 1->01, 2->10, 3->11
    
    outer_terms = [
        # M1: (A00+A11)(B00+B11)
        ([ (1,0), (1,3) ], [ (1,0), (1,3) ]),
        # M2: (A10+A11)B00
        ([ (1,2), (1,3) ], [ (1,0) ]),
        # M3: A00(B01-B11)
        ([ (1,0) ], [ (1,1), (-1,3) ]),
        # M4: A11(B10-B00)
        ([ (1,3) ], [ (1,2), (-1,0) ]),
        # M5: (A00+A01)B11
        ([ (1,0), (1,1) ], [ (1,3) ]),
        # M6: (A10-A00)(B00+B01)
        ([ (1,2), (-1,0) ], [ (1,0), (1,1) ]),
        # M7: (A01-A11)(B10+B11)
        ([ (1,1), (-1,3) ], [ (1,2), (1,3) ]),
    ]
    
    # Inner terms definitions for 2x2:
    # X, Y are 2x2.
    # X00, X01, X10, X11
    # Y00, Y01, Y10, Y11
    
    inner_terms = [
        # s1: (X00+X11)(Y00+Y11)
        ([ (1,0), (1,3) ], [ (1,0), (1,3) ]),
        # s2: (X10+X11)Y00
        ([ (1,2), (1,3) ], [ (1,0) ]),
        # s3: X00(Y01-Y11)
        ([ (1,0) ], [ (1,1), (-1,3) ]),
        # s4: X11(Y10-Y00)
        ([ (1,3) ], [ (1,2), (-1,0) ]),
        # s5: (X00+X01)Y11
        ([ (1,0), (1,1) ], [ (1,3) ]),
        # s6: (X10-X00)(Y00+Y01)
        ([ (1,2), (-1,0) ], [ (1,0), (1,1) ]),
        # s7: (X01-X11)(Y10+Y11)
        ([ (1,1), (-1,3) ], [ (1,2), (1,3) ]),
    ]
    
    # Mapping of outer terms to C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    outer_to_c = [
        [ (1,0), (1,3) ], # M1: C00+1, C11+1
        [ (1,2), (-1,3) ], # M2: C10+1, C11-1
        [ (1,1), (1,3) ], # M3: C01+1, C11+1
        [ (1,0), (1,2) ], # M4: C00+1, C10+1
        [ (-1,0), (1,1) ], # M5: C00-1, C01+1
        [ (1,3) ], # M6: C11+1
        [ (1,0) ], # M7: C00+1
    ]
    
    # Mapping of inner terms to M blocks (2x2 result):
    # Z00 = s1 + s4 - s5 + s7
    # Z01 = s3 + s5
    # Z10 = s2 + s4
    # Z11 = s1 - s2 + s3 + s6
    
    inner_to_m = [
        [ (1,0), (1,3) ], # s1: M00+1, M11+1
        [ (1,2), (-1,3) ], # s2: M10+1, M11-1
        [ (1,1), (1,3) ], # s3: M01+1, M11+1
        [ (1,0), (1,2) ], # s4: M00+1, M10+1
        [ (-1,0), (1,1) ], # s5: M00-1, M01+1
        [ (1,3) ], # s6: M11+1
        [ (1,0) ], # s7: M00+1
    ]
    
    # Helper to map block index and inner index to actual matrix indices.
    # A_block 0:00 -> rows 0,1; cols 0,1
    # A_block 0:01 -> rows 0,1; cols 2,3
    # A_block 0:10 -> rows 2,3; cols 0,1
    # A_block 0:11 -> rows 2,3; cols 2,3
    
    def get_a_indices(block, inner):
        # block: 0..3, inner: 0..3
        # block 0: rows 0,1; block 1: rows 0,1; block 2: rows 2,3; block 3: rows 2,3
        # inner 0: cols 0,1; inner 1: cols 0,1; inner 2: cols 2,3; inner 3: cols 2,3
        # Wait, block is 2x2 of 2x2.
        # block 0 (00): rows 0,1; cols 0,1
        # block 1 (01): rows 0,1; cols 2,3
        # block 2 (10): rows 2,3; cols 0,1
        # block 3 (11): rows 2,3; cols 2,3
        
        # inner 0 (00): row offset 0, col offset 0
        # inner 1 (01): row offset 0, col offset 1
        # inner 2 (10): row offset 1, col offset 0
        # inner 3 (11): row offset 1, col offset 1
        
        row_block = block // 2  # 0 or 1
        col_block = block % 2   # 0 or 1
        inner_row = inner // 2  # 0 or 1
        inner_col = inner % 2   # 0 or 1
        
        r = row_block * 2 + inner_row
        c = col_block * 2 + inner_col
        return r, c
    
    def get_b_indices(block, inner):
        # B is 4x4.
        # block 0 (00): rows 0,1; cols 0,1
        # block 1 (01): rows 0,1; cols 2,3
        # block 2 (10): rows 2,3; cols 0,1
        # block 3 (11): rows 2,3; cols 2,3
        
        row_block = block // 2
        col_block = block % 2
        inner_row = inner // 2
        inner_col = inner % 2
        
        r = row_block * 2 + inner_row
        c = col_block * 2 + inner_col
        return r, c
    
    def get_c_indices(block, inner):
        # C is 4x4.
        row_block = block // 2
        col_block = block % 2
        inner_row = inner // 2
        inner_col = inner % 2
        
        r = row_block * 2 + inner_row
        c = col_block * 2 + inner_col
        return r, c
    
    col = 0
    for m_idx in range(7):
        a_outer = outer_terms[m_idx][0]
        b_outer = outer_terms[m_idx][1]
        c_map = outer_to_c[m_idx]
        
        for s_idx in range(7):
            a_inner = inner_terms[s_idx][0]
            b_inner = inner_terms[s_idx][1]
            m_map = inner_to_m[s_idx]
            
            # Accumulate coefficients for U, V, W
            # U, V, W are vectors of length 16 (flattened 4x4).
            # We need to add the contributions to the global U, V, W columns.
            
            # For each term in a_outer, a_inner, b_outer, b_inner, c_map, m_map:
            # This is a product of sums.
            # Sum_{a_o, a_i, b_o, b_i} (coeff_a_o * coeff_a_i) * A[get_a(a_o, a_i)] * (coeff_b_o * coeff_b_i) * B[get_b(b_o, b_i)] * (coeff_c * coeff_m) * C[get_c(c, m)]
            
            # We iterate over all combinations.
            for coeff_a_o, a_o in a_outer:
                for coeff_a_i, a_i in a_inner:
                    for coeff_b_o, b_o in b_outer:
                        for coeff_b_i, b_i in b_inner:
                            for coeff_c, c_block in c_map:
                                for coeff_m, m_block in m_map:
                                    coeff = coeff_a_o * coeff_a_i * coeff_b_o * coeff_b_i * coeff_c * coeff_m
                                    if coeff != 0:
                                        a_r, a_c = get_a_indices(a_o, a_i)
                                        b_r, b_c = get_b_indices(b_o, b_i)
                                        c_r, c_c = get_c_indices(c_block, m_block)
                                        
                                        u_idx = a_r * 4 + a_c
                                        v_idx = b_r * 4 + b_c
                                        w_idx = c_c * 4 + c_r
                                        
                                        U[u_idx, col] += coeff
                                        V[v_idx, col] += coeff
                                        W[w_idx, col] += coeff
            col += 1
    
    # --- Part 2: 4x4x5 Schoolbook ---
    # A is 4x4.
    # B2 is 4x5 (columns 4..8 of original B).
    # C2 is 4x5 (columns 4..8 of original C).
    
    # Schoolbook: C[i, k] = sum_j A[i, j] B[j, k].
    # For each i, j, k, we have a term.
    # Total terms: 4 * 4 * 5 = 80.
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                u_idx = i * 4 + j
                v_idx = j * 9 + (k + 4)  # B index: row j, col k+4
                w_idx = (k + 4) * 4 + i  # C index: col k+4, row i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
