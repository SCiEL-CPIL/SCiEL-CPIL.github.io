import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Rank 49 for 4x4x4 (Strassen)
    # Rank 48 for 4x4x3 (Schoolbook)
    # Total Rank 97
    rank = 97
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    term_idx = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal term_idx
        for (i, j), val in u_dict.items():
            U[i*m + j, term_idx] += val
        for (j, k), val in v_dict.items():
            V[j*p + k, term_idx] += val
        for (k, i), val in w_dict.items():
            W[k*n + i, term_idx] += val
        term_idx += 1

    # --- Part 1: 4x4x4 Strassen ---
    # We define the 7 Strassen linear forms for the 4x4 blocks.
    # A is 4x4, split into 2x2 blocks A11, A12, A21, A22.
    # B is 4x4, split into 2x2 blocks B11, B12, B21, B22.
    
    # We will iterate through the 7 outer Strassen products M1..M7.
    # For each Mk, we compute the 2x2 product A'_k B'_k using inner Strassen.
    
    # Define the outer Strassen forms.
    # Each form is a dict mapping (block_row, block_col) to coefficient.
    # But we need to map to global (i,j).
    
    # Let's define a helper to get global indices for a block entry.
    # Block (br, bc) entry (r, c) -> global (br*2 + r, bc*2 + c)
    
    # Outer Strassen Forms:
    # M1: A' = A11 + A22, B' = B11 + B22
    # M2: A' = A21 + A22, B' = B11
    # M3: A' = A11, B' = B12 - B22
    # M4: A' = A22, B' = B21 - B11
    # M5: A' = A11 + A12, B' = B22
    # M6: A' = A21 - A11, B' = B11 + B12
    # M7: A' = A12 - A22, B' = B21 + B22
    
    # Inner Strassen Forms for 2x2:
    # P1: a'11 + a'22, b'11 + b'22
    # P2: a'21 + a'22, b'11
    # P3: a'11, b'12 - b'22
    # P4: a'22, b'21 - b'11
    # P5: a'11 + a'12, b'22
    # P6: a'21 - a'11, b'11 + b'12
    # P7: a'12 - a'22, b'21 + b'22
    
    # Contributions of P1..P7 to C'11, C'12, C'21, C'22:
    # C'11: P1(+1), P4(+1), P5(-1), P7(+1)
    # C'12: P3(+1), P5(+1)
    # C'21: P2(+1), P4(+1)
    # C'22: P1(+1), P2(-1), P3(+1), P6(+1)
    
    # We need to map C' entries to global C entries based on the outer Strassen combination.
    # Outer Combination:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each outer term Mk, we know which C block it goes to and with what sign.
    # For each inner term Pm inside Mk, we know which C' entry it contributes to.
    # And we know the global indices of that C' entry.
    
    # Let's pre-define the outer forms as lists of (block, coeff)
    # Block is (br, bc). Entry is (r, c).
    
    outer_forms = [
        # M1
        (
            { (0,0): 1, (1,1): 1 }, # A'
            { (0,0): 1, (1,1): 1 }, # B'
            { (0,0): 1, (1,1): 1 }  # C' targets: C11 and C22
        ),
        # M2
        (
            { (1,0): 1, (1,1): 1 }, # A'
            { (0,0): 1 },           # B'
            { (1,0): 1 }            # C' target: C21
        ),
        # M3
        (
            { (0,0): 1 },           # A'
            { (0,1): 1, (1,1): -1 }, # B'
            { (0,1): 1 }            # C' target: C12
        ),
        # M4
        (
            { (1,1): 1 },           # A'
            { (1,0): 1, (0,0): -1 }, # B'
            { (0,0): 1, (1,0): 1 }   # C' targets: C11 and C21
        ),
        # M5
        (
            { (0,0): 1, (0,1): 1 }, # A'
            { (1,1): 1 },           # B'
            { (0,0): -1, (0,1): 1 } # C' targets: C11 (-1) and C12 (+1)
        ),
        # M6
        (
            { (1,0): 1, (0,0): -1 }, # A'
            { (0,0): 1, (0,1): 1 }, # B'
            { (1,1): 1 }            # C' target: C22
        ),
        # M7
        (
            { (0,1): 1, (1,1): -1 }, # A'
            { (1,0): 1, (1,1): 1 }, # B'
            { (0,0): 1, (0,1): 1 }  # C' targets: C11 and C12
        ),
    ]
    
    # Inner Strassen forms for 2x2.
    # These operate on local indices 0,1.
    inner_forms = [
        # P1
        (
            { (0,0): 1, (1,1): 1 }, # a'
            { (0,0): 1, (1,1): 1 }, # b'
            { (0,0): 1, (1,1): 1 }  # c' contributions: C11(+1), C22(+1)
        ),
        # P2
        (
            { (1,0): 1, (1,1): 1 }, # a'
            { (0,0): 1 },           # b'
            { (1,0): 1, (1,1): -1 } # c' contributions: C21(+1), C22(-1)
        ),
        # P3
        (
            { (0,0): 1 },           # a'
            { (0,1): 1, (1,1): -1 }, # b'
            { (0,1): 1, (1,1): 1 }   # c' contributions: C12(+1), C22(+1)
        ),
        # P4
        (
            { (1,1): 1 },           # a'
            { (1,0): 1, (0,0): -1 }, # b'
            { (0,0): 1, (1,0): 1 }   # c' contributions: C11(+1), C21(+1)
        ),
        # P5
        (
            { (0,0): 1, (0,1): 1 }, # a'
            { (1,1): 1 },           # b'
            { (0,0): -1, (0,1): 1 } # c' contributions: C11(-1), C12(+1)
        ),
        # P6
        (
            { (1,0): 1, (0,0): -1 }, # a'
            { (0,0): 1, (0,1): 1 }, # b'
            { (1,1): 1 }            # c' contribution: C22(+1)
        ),
        # P7
        (
            { (0,1): 1, (1,1): -1 }, # a'
            { (1,0): 1, (1,1): 1 }, # b'
            { (0,0): 1, (0,1): 1 }  # c' contributions: C11(+1), C12(+1)
        ),
    ]
    
    # Wait, the inner forms' c' contributions are fixed for a 2x2 product.
    # The outer forms determine WHERE that 2x2 product lands in the 4x4 C.
    
    # Let's refine the logic.
    # For each outer term k (0..6):
    #   Get A'_k blocks, B'_k blocks, and C'_k target blocks with signs.
    #   For each inner term m (0..6):
    #     Get a'_m local entries, b'_m local entries, and c'_m local entries with signs.
    #     Construct global U, V, W dicts.
    
    for k in range(7):
        A_blocks, B_blocks, C_targets = outer_forms[k]
        
        # Expand A_blocks and B_blocks to global (i,j) and (j,k) dicts.
        # But we need to combine them with the inner forms.
        # It's easier to iterate inner forms and combine.
        
        for m in range(7):
            a_local, b_local, c_local_contribs = inner_forms[m]
            
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Combine A_outer and A_inner
            # A_outer is dict of (br, bc) -> coeff
            # A_inner is dict of (r, c) -> coeff
            # Global (i, j) = (br*2 + r, bc*2 + c)
            # Coeff is product of outer and inner coeffs.
            
            for (br, bc), coeff_a_out in A_blocks.items():
                for (r, c), coeff_a_in in a_local.items():
                    i = br*2 + r
                    j = bc*2 + c
                    u_dict[(i, j)] = u_dict.get((i, j), 0) + coeff_a_out * coeff_a_in
            
            for (br, bc), coeff_b_out in B_blocks.items():
                for (r, c), coeff_b_in in b_local.items():
                    j = br*2 + r
                    k_idx = bc*2 + c
                    v_dict[(j, k_idx)] = v_dict.get((j, k_idx), 0) + coeff_b_out * coeff_b_in
            
            # Combine C_outer targets and C_inner contributions
            # C_outer is dict of (br, bc) -> coeff (sign)
            # C_inner is dict of (r, c) -> coeff (sign)
            # Global C index is (k, i) where k is col, i is row.
            # Wait, W index is (k, i).
            # C block (br, bc) has rows br*2..br*2+1 and cols bc*2..bc*2+1.
            # Entry (r, c) in block is global row br*2+r, col bc*2+c.
            # So W index is (col, row) = (bc*2+c, br*2+r).
            
            for (br, bc), coeff_c_out in C_targets.items():
                for (r, c), coeff_c_in in c_local_contribs.items():
                    row = br*2 + r
                    col = bc*2 + c
                    w_dict[(col, row)] = w_dict.get((col, row), 0) + coeff_c_out * coeff_c_in
            
            add_term(u_dict, v_dict, w_dict)

    # --- Part 2: 4x4x3 Schoolbook ---
    # C_{i,k} for k=4,5,6 (0-based: 4,5,6)
    # Actually B is 4x7. Cols 0-3 are in Part 1. Cols 4-6 are in Part 2.
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_dict = {(i, j): 1.0}
                v_dict = {(j, k): 1.0}
                w_dict = {(k, i): 1.0}
                add_term(u_dict, v_dict, w_dict)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
