import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
      - Part 1: 4x4 * 4x4 (B columns 0..3) using Strassen, rank 49.
      - Part 2: 4x4 * 4x3 (B columns 4..6) using Schoolbook, rank 48.
      - Total rank = 97.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 110

    terms = []

    # ------------------------------------------------------------------
    # Part 1: Strassen for 4x4 * 4x4 (Rank 49)
    # ------------------------------------------------------------------
    # Strassen rank-7 decomposition of 2x2x2 matmul tensor.
    # s2x2 list of (xvec, yvec, zvec) length 4 tuples (00,01,10,11)
    s2x2 = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, -1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (0, -1, 0, 1)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 0, 0)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (-1, 0, 0, 0)),
    ]

    # Define the 7 Strassen products M1..M7 for the 4x4 case.
    # Each product is defined by:
    #   x_blocks: list of (block_row, block_col, coeff) for A
    #   y_blocks: list of (block_row, block_col, coeff) for B
    #   c_contrib: list of (c_block_row, c_block_col, coeff) for C
    # Note: The "block" here refers to the 2x2 sub-blocks of the 4x4 matrices.
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)

    strassen_defs = [
        # M1 = (A00+A11)*(B00+B11)
        {
            'x': [((0, 0), 1), ((1, 1), 1)],
            'y': [((0, 0), 1), ((1, 1), 1)],
            'c': [((0, 0), 1), ((1, 1), 1)]
        },
        # M2 = (A10+A11)*B00
        {
            'x': [((1, 0), 1), ((1, 1), 1)],
            'y': [((0, 0), 1)],
            'c': [((1, 0), 1), ((1, 1), -1)]
        },
        # M3 = A00*(B01-B11)
        {
            'x': [((0, 0), 1)],
            'y': [((0, 1), 1), ((1, 1), -1)],
            'c': [((0, 1), 1), ((1, 1), 1)]
        },
        # M4 = A11*(B10-B00)
        {
            'x': [((1, 1), 1)],
            'y': [((1, 0), 1), ((0, 0), -1)],
            'c': [((0, 0), 1), ((1, 0), 1)]
        },
        # M5 = (A00+A01)*B11
        {
            'x': [((0, 0), 1), ((0, 1), 1)],
            'y': [((1, 1), 1)],
            'c': [((0, 1), -1), ((1, 1), 1)]
        },
        # M6 = (A10-A00)*(B00+B01)
        {
            'x': [((1, 0), 1), ((0, 0), -1)],
            'y': [((0, 0), 1), ((0, 1), 1)],
            'c': [((0, 0), 1)]
        },
        # M7 = (A01-A11)*(B10+B11)
        {
            'x': [((0, 1), 1), ((1, 1), -1)],
            'y': [((1, 0), 1), ((1, 1), 1)],
            'c': [((0, 0), -1), ((1, 1), 1)]
        },
    ]

    for def_idx, sdef in enumerate(strassen_defs):
        # Build the 2x2 matrices X and Y representing the linear combinations of A and B blocks.
        # X[local_r, local_c] is a linear combination of A global entries.
        # We represent the contribution of each global A entry (ar, ac) to the local 2x2 matrix X.
        # Since the Strassen algorithm operates on the 2x2 matrices X and Y, we need to map
        # the global A entries to the "virtual" 2x2 matrix X.
        
        # For each of the 7 Strassen steps (t in 0..6):
        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            
            a_vec = {}
            b_vec = {}
            c_vec = {}
            
            # Process X (from A)
            # xvec is (x00, x01, x10, x11)
            # These correspond to the entries of the 2x2 matrix X.
            # X[0,0] coefficient is xvec[0], etc.
            # But X itself is formed from A blocks.
            # We need to determine the global A entries that contribute to X[local_r, local_c].
            
            # Let's build the mapping from global A (ar, ac) to local X (lr, lc).
            # For each block (br, bc) with coeff c in sdef['x']:
            #   The block A[br*2:br*2+2, bc*2:bc*2+2] contributes c * A[...] to X.
            #   Specifically, A[br*2+lr, bc*2+lc] contributes c to X[lr, lc].
            
            x_map = {} # (ar, ac) -> { (lr, lc) -> coeff }
            for (br, bc), coeff in sdef['x']:
                for lr in range(2):
                    for lc in range(2):
                        ar = br * 2 + lr
                        ac = bc * 2 + lc
                        if (ar, ac) not in x_map:
                            x_map[(ar, ac)] = {}
                        x_map[(ar, ac)][(lr, lc)] = coeff

            # Now, for each global A entry (ar, ac), we know which local X entries it affects.
            # The Strassen term t uses xvec[lr*2+lc] as the coefficient for X[lr, lc].
            # So the total coefficient for A[ar, ac] in the a_vec is sum over (lr, lc) of 
            # x_map[(ar, ac)].get((lr, lc), 0) * xvec[lr*2+lc].
            
            for (ar, ac), local_coeffs in x_map.items():
                total_coeff = 0
                for (lr, lc), c_val in local_coeffs.items():
                    total_coeff += c_val * xvec[lr * 2 + lc]
                if total_coeff != 0:
                    a_vec[(ar, ac)] = a_vec.get((ar, ac), 0) + total_coeff

            # Process Y (from B)
            y_map = {} # (br, bc) -> { (lr, lc) -> coeff }
            for (br, bc), coeff in sdef['y']:
                for lr in range(2):
                    for lc in range(2):
                        br_g = br * 2 + lr
                        bc_g = bc * 2 + lc
                        if (br_g, bc_g) not in y_map:
                            y_map[(br_g, bc_g)] = {}
                        y_map[(br_g, bc_g)][(lr, lc)] = coeff

            for (br_g, bc_g), local_coeffs in y_map.items():
                total_coeff = 0
                for (lr, lc), c_val in local_coeffs.items():
                    total_coeff += c_val * yvec[lr * 2 + lc]
                if total_coeff != 0:
                    b_vec[(br_g, bc_g)] = b_vec.get((br_g, bc_g), 0) + total_coeff

            # Process Z (to C)
            # zvec is (z00, z01, z10, z11)
            # These correspond to the entries of the 2x2 matrix Z = M_k.
            # M_k contributes to C blocks according to sdef['c'].
            # For each C block (cr, cc) with coeff c_c in sdef['c']:
            #   M_k[lr, lc] contributes c_c * M_k[lr, lc] to C[cr*2+lr, cc*2+lc].
            
            # We need to map the local Z entries (lr, lc) to global C entries (cr_g, cc_g).
            # The Strassen term t uses zvec[lr*2+lc] as the coefficient for Z[lr, lc].
            # So the total coefficient for C[cr_g, cc_g] is sum over (cr, cc) in sdef['c']
            # and (lr, lc) such that cr_g = cr*2+lr, cc_g = cc*2+lc.
            
            # Let's iterate over sdef['c'] and zvec.
            for (cr, cc), c_coeff in sdef['c']:
                for lr in range(2):
                    for lc in range(2):
                        cr_g = cr * 2 + lr
                        cc_g = cc * 2 + lc
                        # The local Z entry is Z[lr, lc]
                        # Its coefficient in the Strassen product is zvec[lr*2+lc]
                        # Its contribution to C is c_coeff
                        val = c_coeff * zvec[lr * 2 + lc]
                        if val != 0:
                            c_vec[(cr_g, cc_g)] = c_vec.get((cr_g, cc_g), 0) + val

            if a_vec and b_vec and c_vec:
                terms.append((a_vec, b_vec, c_vec))

    # ------------------------------------------------------------------
    # Part 2: Schoolbook for 4x4 * 4x3 (Rank 48)
    # ------------------------------------------------------------------
    # B columns 4, 5, 6
    for i in range(4): # Row of A
        for k in range(4): # Col of A / Row of B
            for j in range(4, 7): # Col of B
                a_vec = {(i, k): 1}
                b_vec = {(k, j): 1}
                c_vec = {(i, j): 1}
                terms.append((a_vec, b_vec, c_vec))

    rank = len(terms)

    # Build U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r, (a_vec, b_vec, c_vec) in enumerate(terms):
        for (ar, ac), val in a_vec.items():
            U[ar * m + ac][r] = val
        for (br, bc), val in b_vec.items():
            V[br * p + bc][r] = val
        for (cr, cc), val in c_vec.items():
            W[cc * n + cr][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
