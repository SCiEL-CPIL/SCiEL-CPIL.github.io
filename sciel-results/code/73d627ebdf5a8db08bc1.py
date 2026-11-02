import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: Rank 49 via Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to generate the 49 rank-1 terms.
    # Term index r from 0 to 48.
    # r = t * 7 + s, where t is the 4x4 Strassen step (0-6), s is the 2x2 Strassen step (0-6).
    
    # Define the 2x2 Strassen coefficients for U, V, and W contributions.
    # For 2x2 matrices A, B:
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # The 7 multiplications M1..M7:
    # Each M is a product of linear forms in A and B.
    # We store the coefficients for A (u), B (v), and the contribution to C (w).
    # Note: w is a vector of length 4 (for 2x2 C).
    
    # M1: (A00+A11)(B00+B11)
    # u: [1, 0, 0, 1] (A00, A01, A10, A11) -> indices 0, 3
    # v: [1, 0, 0, 1]
    # M1 contributes to C00 (+1), C11 (+1).
    # w: [1, 0, 0, 1] (C00, C01, C10, C11) -> indices 0, 3
    
    # M2: (A10+A11)B00
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # M2 contributes to C10 (+1), C11 (-1).
    # w: [0, 0, 1, -1]
    
    # M3: A00(B01-B11)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # M3 contributes to C01 (+1), C11 (+1).
    # w: [0, 1, 0, 1]
    
    # M4: A11(B10-B00)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1] # B10 is idx 2, B00 is idx 0. B10-B00 -> [ -1, 0, 1, 0 ]?
    # Let's check indices: B00=0, B01=1, B10=2, B11=3.
    # B10 - B00 -> v = [-1, 0, 1, 0].
    # M4 contributes to C00 (+1), C10 (+1).
    # w: [1, 0, 1, 0]
    
    # M5: (A00+A01)B11
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # M5 contributes to C00 (-1), C01 (+1).
    # w: [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B01)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # M6 contributes to C11 (+1).
    # w: [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # M7 contributes to C00 (+1).
    # w: [1, 0, 0, 0]
    
    # List of (u_coeffs, v_coeffs, w_coeffs) for the 7 2x2 steps.
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Now map these to the 4x4 level.
    # The 4x4 Strassen steps t=0..6 correspond to the 7 multiplications M1..M7.
    # For each t, we need to map the 2x2 indices (0..3) to the 4x4 indices (0..15).
    # The blocks are:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For a given t, the inputs to the 2x2 Strassen are linear combinations of the 4 blocks.
    # We define the linear forms for the 4 blocks of A and B for each t.
    # Let A_blocks be a 4x4 matrix of coefficients? No, just 4 vectors of length 16.
    # For each t, we have 4 linear forms for A (corresponding to A00, A01, A10, A11 of the 2x2 view).
    # And 4 linear forms for B.
    
    # t=0 (M1): X = A00 + A11, Y = B00 + B11
    # X00 = A00, X01 = A01? No.
    # M1 = (A00 + A11) (B00 + B11).
    # Let X = A00 + A11. X is a 2x2 matrix.
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    
    # So for t=0:
    # u_block_0 (X00) = A00_00 + A11_00
    # u_block_1 (X01) = A00_01 + A11_01
    # u_block_2 (X10) = A00_10 + A11_10
    # u_block_3 (X11) = A00_11 + A11_11
    
    # We can represent this as:
    # For each of the 4 "virtual" 2x2 indices (0..3), we have a linear form over the 16 elements of A.
    
    # Let's define the mapping for each t.
    # t=0: X = A00+A11, Y = B00+B11
    # t=1: X = A10+A11, Y = B00
    # t=2: X = A00, Y = B01-B11
    # t=3: X = A11, Y = B10-B00
    # t=4: X = A00+A01, Y = B11
    # t=5: X = A10-A00, Y = B00+B01
    # t=6: X = A01-A11, Y = B10+B11
    
    # For each t, we define 4 linear forms for A and 4 for B.
    # Each linear form is a vector of length 16.
    
    def get_block_linear_form(block_idx, coeff):
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns a vector of length 16 with `coeff` at the indices of the block.
        vec = [0] * 16
        if block_idx == 0: # A00
            vec[0] = coeff; vec[1] = coeff; vec[4] = coeff; vec[5] = coeff
        elif block_idx == 1: # A01
            vec[2] = coeff; vec[3] = coeff; vec[6] = coeff; vec[7] = coeff
        elif block_idx == 2: # A10
            vec[8] = coeff; vec[9] = coeff; vec[12] = coeff; vec[13] = coeff
        elif block_idx == 3: # A11
            vec[10] = coeff; vec[11] = coeff; vec[14] = coeff; vec[15] = coeff
        return vec

    def add_vec(v1, v2, sign=1):
        return [v1[i] + sign * v2[i] for i in range(16)]

    # Define the linear forms for A and B for each t.
    # A_forms[t][k] is the linear form for the k-th element of the 2x2 matrix X.
    # B_forms[t][k] is the linear form for the k-th element of the 2x2 matrix Y.
    
    A_forms = []
    B_forms = []
    
    # t=0
    A_forms.append([
        add_vec(get_block_linear_form(0, 1), get_block_linear_form(3, 1)), # X00
        add_vec(get_block_linear_form(0, 1), get_block_linear_form(3, 1)), # X01? No.
        # X = A00 + A11.
        # X[0,0] = A00[0,0] + A11[0,0]
        # X[0,1] = A00[0,1] + A11[0,1]
        # The linear form for X[0,1] is A00_01 + A11_01.
        # get_block_linear_form(0, 1) has 1s at A00 indices.
        # We need to distinguish which element of the block.
        # My get_block_linear_form returns a vector with 1s at ALL indices of the block.
        # This is incorrect. X[0,0] uses only A00[0,0].
        # So I need to create linear forms for specific elements.
    ])
    
    # Let's refine get_element_linear_form(block_idx, row, col, coeff)
    def get_elem_linear_form(block_idx, r, c, coeff):
        vec = [0] * 16
        # Base index for block
        base = block_idx * 4 # 0, 4, 8, 12? No.
        # A00: 0,1,4,5. Indices: r*4+c? No.
        # Block 0 (A00): rows 0-1, cols 0-1.
        # Index in A: row*4 + col.
        # A00[0,0] -> 0. A00[0,1] -> 1. A00[1,0] -> 4. A00[1,1] -> 5.
        # A01[0,0] -> 2. A01[0,1] -> 3. A01[1,0] -> 6. A01[1,1] -> 7.
        # A10[0,0] -> 8. A10[0,1] -> 9. A10[1,0] -> 12. A10[1,1] -> 13.
        # A11[0,0] -> 10. A11[0,1] -> 11. A11[1,0] -> 14. A11[1,1] -> 15.
        
        if block_idx == 0:
            idx = r * 4 + c
        elif block_idx == 1:
            idx = r * 4 + (c + 2)
        elif block_idx == 2:
            idx = (r + 2) * 4 + c
        elif block_idx == 3:
            idx = (r + 2) * 4 + (c + 2)
        vec[idx] = coeff
        return vec

    # Re-define A_forms and B_forms
    A_forms = []
    B_forms = []
    
    # t=0: X = A00+A11, Y = B00+B11
    # X[r,c] = A00[r,c] + A11[r,c]
    A_forms.append([
        [add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(3, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    # Flatten to list of 4 vectors
    A_forms[0] = [A_forms[0][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(3, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    B_forms[0] = [B_forms[0][r][c] for r in range(2) for c in range(2)]
    
    # t=1: X = A10+A11, Y = B00
    # X[r,c] = A10[r,c] + A11[r,c]
    A_forms.append([
        [add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(3, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    A_forms[1] = [A_forms[1][r][c] for r in range(2) for c in range(2)]
    
    # Y = B00. Y[r,c] = B00[r,c]
    B_forms.append([
        [get_elem_linear_form(0, r, c, 1) for c in range(2)] for r in range(2)
    ])
    B_forms[1] = [B_forms[1][r][c] for r in range(2) for c in range(2)]
    
    # t=2: X = A00, Y = B01-B11
    A_forms.append([
        [get_elem_linear_form(0, r, c, 1) for c in range(2)] for r in range(2)
    ])
    A_forms[2] = [A_forms[2][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [add_vec(get_elem_linear_form(1, r, c, 1), get_elem_linear_form(3, r, c, -1)) for c in range(2)] for r in range(2)
    ])
    B_forms[2] = [B_forms[2][r][c] for r in range(2) for c in range(2)]
    
    # t=3: X = A11, Y = B10-B00
    A_forms.append([
        [get_elem_linear_form(3, r, c, 1) for c in range(2)] for r in range(2)
    ])
    A_forms[3] = [A_forms[3][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(0, r, c, -1)) for c in range(2)] for r in range(2)
    ])
    B_forms[3] = [B_forms[3][r][c] for r in range(2) for c in range(2)]
    
    # t=4: X = A00+A01, Y = B11
    A_forms.append([
        [add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(1, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    A_forms[4] = [A_forms[4][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [get_elem_linear_form(3, r, c, 1) for c in range(2)] for r in range(2)
    ])
    B_forms[4] = [B_forms[4][r][c] for r in range(2) for c in range(2)]
    
    # t=5: X = A10-A00, Y = B00+B01
    A_forms.append([
        [add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(0, r, c, -1)) for c in range(2)] for r in range(2)
    ])
    A_forms[5] = [A_forms[5][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(1, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    B_forms[5] = [B_forms[5][r][c] for r in range(2) for c in range(2)]
    
    # t=6: X = A01-A11, Y = B10+B11
    A_forms.append([
        [add_vec(get_elem_linear_form(1, r, c, 1), get_elem_linear_form(3, r, c, -1)) for c in range(2)] for r in range(2)
    ])
    A_forms[6] = [A_forms[6][r][c] for r in range(2) for c in range(2)]
    
    B_forms.append([
        [add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(3, r, c, 1)) for c in range(2)] for r in range(2)
    ])
    B_forms[6] = [B_forms[6][r][c] for r in range(2) for c in range(2)]

    # Now generate the 49 terms.
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            # Get the 2x2 coefficients for this step s
            u_2x2, v_2x2, w_2x2 = strassen_2x2[s]
            
            # Construct full u, v vectors (length 16)
            u_full = [0] * 16
            v_full = [0] * 16
            
            # Sum the contributions from the 4 blocks
            for k in range(4):
                coeff_u = u_2x2[k]
                coeff_v = v_2x2[k]
                if coeff_u != 0:
                    u_full = add_vec(u_full, A_forms[t][k], coeff_u)
                if coeff_v != 0:
                    v_full = add_vec(v_full, B_forms[t][k], coeff_v)
            
            # Construct w_full vector (length 16)
            # w_2x2 has 4 entries corresponding to C00, C01, C10, C11 of the 2x2 result M_t.
            # M_t is a 2x2 matrix.
            # M_t[0,0] contributes to C blocks.
            # We need to know which C blocks M_t contributes to.
            
            # M1 (t=0): C00, C11
            # M2 (t=1): C10, C11
            # M3 (t=2): C01, C11
            # M4 (t=3): C00, C10
            # M5 (t=4): C00, C01
            # M6 (t=5): C11
            # M7 (t=6): C00
            
            # For each k (0..3) in w_2x2:
            # k=0 (M_t[0,0]): contributes to C_block_row, C_block_col
            # k=1 (M_t[0,1]): ...
            
            # Define the block mapping for each t.
            # t=0: blocks (0,0) and (1,1)
            # t=1: blocks (1,0) and (1,1)
            # t=2: blocks (0,1) and (1,1)
            # t=3: blocks (0,0) and (1,0)
            # t=4: blocks (0,0) and (0,1)
            # t=5: blocks (1,1)
            # t=6: blocks (0,0)
            
            w_full = [0] * 16
            
            # Mapping of t to list of (block_row, block_col, weight)
            # The weight is the coefficient from the 4x4 Strassen formula.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So:
            # M1 (t=0): C00 (+1), C11 (+1)
            # M2 (t=1): C10 (+1), C11 (-1)
            # M3 (t=2): C01 (+1), C11 (+1)
            # M4 (t=3): C00 (+1), C10 (+1)
            # M5 (t=4): C00 (-1), C01 (+1)
            # M6 (t=5): C11 (+1)
            # M7 (t=6): C00 (+1)
            
            block_contribs = {
                0: [(0, 0, 1), (1, 1, 1)],
                1: [(1, 0, 1), (1, 1, -1)],
                2: [(0, 1, 1), (1, 1, 1)],
                3: [(0, 0, 1), (1, 0, 1)],
                4: [(0, 0, -1), (0, 1, 1)],
                5: [(1, 1, 1)],
                6: [(0, 0, 1)]
            }
            
            for k in range(4):
                coeff_w = w_2x2[k]
                if coeff_w == 0:
                    continue
                # k corresponds to M_t[r, c] where r = k//2, c = k%2
                r_block = k // 2
                c_block = k % 2
                
                for (br, bc, weight) in block_contribs[t]:
                    # The element M_t[r_block, c_block] adds to C[br*2 + r_block, bc*2 + c_block]
                    # With weight `weight` from the 4x4 formula.
                    # And `coeff_w` from the 2x2 formula.
                    # Total coeff = weight * coeff_w.
                    total_coeff = weight * coeff_w
                    if total_coeff != 0:
                        row = br * 2 + r_block
                        col = bc * 2 + c_block
                        # W index: col * 4 + row
                        idx = col * 4 + row
                        w_full[idx] += total_coeff
            
            # Assign to U, V, W
            for i in range(16):
                U[i][r] = u_full[i]
                V[i][r] = v_full[i]
                W[i][r] = w_full[i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
