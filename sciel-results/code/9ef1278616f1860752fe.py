import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Returns list of 7 tuples: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping local A index (0..3) to coeff
    # V_coeffs: dict mapping local B index (0..3) to coeff
    # W_coeffs: dict mapping local C index (0..3) to coeff
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {}))
        # M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {}))
        # M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {}))
        # M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {}))
        # M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {}))
        # M7
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {}))
        
        # Now we need to map these to C contributions
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will merge these into the W coefficients for each term
        # Actually, it's easier to just define the 7 terms and then add to W
        # But the decomposition requires each term to be rank 1.
        # So W for term t must contain all contributions of Mt to C.
        
        # Let's re-define terms with full W coefficients
        
        # Term 0 (M1): A(0,3) B(0,3) -> C00(+1), C11(+1)
        terms[0] = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        
        # Term 1 (M2): A(2,3) B(0) -> C10(+1)
        terms[1] = ({2: 1, 3: 1}, {0: 1}, {2: 1}) # C10 is index 2 (row 1, col 0)
        
        # Term 2 (M3): A(0) B(1,3) -> C01(+1)
        terms[2] = ({0: 1}, {1: 1, 3: -1}, {1: 1}) # C01 is index 1 (row 0, col 1)
        
        # Term 3 (M4): A(3) B(2,0) -> C00(+1), C10(+1)
        terms[3] = ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        
        # Term 4 (M5): A(0,1) B(3) -> C00(-1), C01(+1)
        terms[4] = ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        
        # Term 5 (M6): A(2,0) B(0,1) -> C11(+1)
        terms[5] = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        
        # Term 6 (M7): A(1,3) B(2,3) -> C00(+1), C11(+1)
        terms[6] = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1})
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Define the 8 blocks
    # Each block: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4, B is 4x8, C is 2x8
    # A indices: r in 0..1, c in 0..3
    # B indices: r in 0..3, c in 0..7
    # C indices: r in 0..1, c in 0..7
    
    blocks = []
    # Left half of C (cols 0-3)
    # C cols 0,1 come from B cols 0,1
    # C cols 2,3 come from B cols 2,3
    # A cols 0,1 and 2,3
    
    # Block 1: A(0:2, 0:2) * B(0:2, 0:2) -> C(0:2, 0:2)
    blocks.append((0, 0, 0, 0, 0, 0))
    # Block 2: A(0:2, 2:4) * B(2:4, 0:2) -> C(0:2, 0:2)
    blocks.append((0, 2, 2, 0, 0, 0))
    # Block 3: A(0:2, 0:2) * B(0:2, 2:4) -> C(0:2, 2:4)
    blocks.append((0, 0, 0, 2, 0, 2))
    # Block 4: A(0:2, 2:4) * B(2:4, 2:4) -> C(0:2, 2:4)
    blocks.append((0, 2, 2, 2, 0, 2))
    
    # Right half of C (cols 4-7)
    # C cols 4,5 come from B cols 4,5
    # C cols 6,7 come from B cols 6,7
    
    # Block 5: A(0:2, 0:2) * B(0:2, 4:6) -> C(0:2, 4:6)
    blocks.append((0, 0, 0, 4, 0, 4))
    # Block 6: A(0:2, 2:4) * B(2:4, 4:6) -> C(0:2, 4:6)
    blocks.append((0, 2, 2, 4, 0, 4))
    # Block 7: A(0:2, 0:2) * B(0:2, 6:8) -> C(0:2, 6:8)
    blocks.append((0, 0, 0, 6, 0, 6))
    # Block 8: A(0:2, 2:4) * B(2:4, 6:8) -> C(0:2, 6:8)
    blocks.append((0, 2, 2, 6, 0, 6))
    
    term_idx = 0
    for a_r, a_c, b_r, b_c, c_r, c_c in blocks:
        for u_dict, v_dict, w_dict in strassen_terms:
            # Fill U
            for local_idx, coeff in u_dict.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = a_r + r
                global_c = a_c + c
                global_idx = global_r * m + global_c
                U[global_idx][term_idx] = coeff
                
            # Fill V
            for local_idx, coeff in v_dict.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = b_r + r
                global_c = b_c + c
                global_idx = global_r * p + global_c
                V[global_idx][term_idx] = coeff
                
            # Fill W
            for local_idx, coeff in w_dict.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = c_r + r
                global_c = c_c + c
                # W index: k*n + i -> global_c * n + global_r
                global_idx = global_c * n + global_r
                W[global_idx][term_idx] = coeff
                
            term_idx += 1
            
    return {"U": U, "V": V, "W": W}
