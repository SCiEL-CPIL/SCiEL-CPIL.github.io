import time

def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication
    using 2-level Strassen's algorithm.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0.0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0.0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0.0] * rank for _ in range(p * n)]  # (16, 49)

    # Each scalar multiplication is identified by (k, s) where k=0..6 (top Strassen), s=0..6 (inner Strassen)
    # rank index r = 7*k + s

    # For each top-level Strassen product k, we have:
    #   X_k: 2x2 matrix of linear forms in A (each entry is a set of (row,col) indices with coeff 1 or -1)
    #   Y_k: 2x2 matrix of linear forms in B
    #   D_k: 2x2 matrix showing how the 2x2 result contributes to C blocks
    #   C_k: dict mapping (block_row, block_col) -> coefficient

    # Top-level Strassen definitions
    # Each X and Y entry is a list of (global_row, global_col, coeff) for A or B
    # For a 2x2 block at (br, bc), entry (r,c) maps to global (2*br+r, 2*bc+c)

    def block_entries(br, bc):
        """Return 2x2 list of (global_i, global_j) for block (br,bc)"""
        return [[(2*br+r, 2*bc+c) for c in range(2)] for r in range(2)]

    # Define linear forms as dicts: {(global_i, global_j): coeff}
    def single_block_form(br, bc):
        """2x2 list of linear forms for A_{br,bc}"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br+r, 2*bc+c): 1})
            res.append(row)
        return res

    def sum_block_form(br1, bc1, br2, bc2):
        """2x2 list of linear forms for A_{br1,bc1} + A_{br2,bc2}"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br1+r, 2*bc1+c): 1, (2*br2+r, 2*bc2+c): 1})
            res.append(row)
        return res

    def diff_block_form(br1, bc1, br2, bc2):
        """2x2 list of linear forms for A_{br1,bc1} - A_{br2,bc2}"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br1+r, 2*bc1+c): 1, (2*br2+r, 2*bc2+c): -1})
            res.append(row)
        return res

    # Top-level Strassen products
    # k=0: M0 = (A00+A11)(B00+B11)
    # k=1: M1 = (A10+A11)B00
    # k=2: M2 = A00(B01-B11)
    # k=3: M3 = A11(B10-B00)
    # k=4: M4 = (A00+A01)B11
    # k=5: M5 = (A10-A00)(B00+B01)
    # k=6: M6 = (A01-A11)(B10+B11)

    # C reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    top_products = [
        # (X_form_type, X_params, Y_form_type, Y_params, C_contrib)
        # C_contrib: {(block_r, block_c): coeff}
        ({'x': 'sum', 'p1': (0,0), 'p2': (1,1)}, 'A',
         {'y': 'sum', 'p1': (0,0), 'p2': (1,1)}, 'B',
         {(0,0): 1, (1,1): 1}),

        ({'x': 'sum', 'p1': (1,0), 'p2': (1,1)}, 'A',
         {'y': 'single', 'p': (0,0)}, 'B',
         {(1,0): 1, (1,1): -1}),

        ({'x': 'single', 'p': (0,0)}, 'A',
         {'y': 'diff', 'p1': (0,1), 'p2': (1,1)}, 'B',
         {(0,1): 1, (1,1): 1}),

        ({'x': 'single', 'p': (1,1)}, 'A',
         {'y': 'diff', 'p1': (1,0), 'p2': (0,0)}, 'B',
         {(0,0): 1, (1,0): 1}),

        ({'x': 'sum', 'p1': (0,0), 'p2': (0,1)}, 'A',
         {'y': 'single', 'p': (1,1)}, 'B',
         {(0,0): -1, (0,1): 1}),

        ({'x': 'diff', 'p1': (1,0), 'p2': (0,0)}, 'A',
         {'y': 'sum', 'p1': (0,0), 'p2': (0,1)}, 'B',
         {(1,1): 1}),

        ({'x': 'diff', 'p1': (0,1), 'p2': (1,1)}, 'A',
         {'y': 'sum', 'p1': (1,0), 'p2': (1,1)}, 'B',
         {(0,0): 1}),
    ]

    # Inner Strassen: for a 2x2 product P = X * Y where X,Y are 2x2
    # The 7 Strassen multiplies for 2x2:
    # s=0: z0 = (X00+X11)(Y00+Y11)
    # s=1: z1 = (X10+X11)Y00
    # s=2: z2 = X00(Y01-Y11)
    # s=3: z3 = X11(Y10-Y00)
    # s=4: z4 = (X00+X01)Y11
    # s=5: z5 = (X10-X00)(Y00+Y01)
    # s=6: z6 = (X01-X11)(Y10+Y11)
    #
    # Result:
    # P00 = z0 + z3 - z4 + z6
    # P01 = z2 + z4
    # P10 = z1 + z3
    # P11 = z0 - z1 + z2 + z5

    # For each inner Strassen step s, the scalar multiplication is:
    #   u_s = linear form in X (2x2 matrix of linear forms in A)
    #   v_s = linear form in Y (2x2 matrix of linear forms in B)
    #   The product u_s * v_s contributes to P at positions given by D_s
    #   D_s is a 2x2 matrix of coefficients

    # D_s matrices (how z_s contributes to P entries):
    D = [
        [[1, 0], [0, 1]],   # s=0: z0 -> P00 += 1, P11 += 1
        [[0, 0], [1, -1]],  # s=1: z1 -> P10 += 1, P11 -= 1
        [[0, 1], [0, 1]],   # s=2: z2 -> P01 += 1, P11 += 1
        [[1, 0], [1, 0]],   # s=3: z3 -> P00 += 1, P10 += 1
        [[-1, 1], [0, 0]],  # s=4: z4 -> P00 -= 1, P01 += 1
        [[0, 0], [0, 1]],   # s=5: z5 -> P11 += 1
        [[1, 0], [0, 0]],   # s=6: z6 -> P00 += 1
    ]

    # For each inner Strassen step s, determine which entries of X and Y are used
    # with what coefficients:
    # s=0: u = X00+X11, v = Y00+Y11
    # s=1: u = X10+X11, v = Y00
    # s=2: u = X00, v = Y01-Y11
    # s=3: u = X11, v = Y10-Y00
    # s=4: u = X00+X01, v = Y11
    # s=5: u = X10-X00, v = Y00+Y01
    # s=6: u = X01-X11, v = Y10+Y11

    # u_entries[s] = list of (local_r, local_c, coeff) for X
    # v_entries[s] = list of (local_r, local_c, coeff) for Y
    u_entries = [
        [(0,0,1), (1,1,1)],           # s=0
        [(1,0,1), (1,1,1)],           # s=1
        [(0,0,1)],                     # s=2
        [(1,1,1)],                     # s=3
        [(0,0,1), (0,1,1)],           # s=4
        [(1,0,1), (0,0,-1)],          # s=5
        [(0,1,1), (1,1,-1)],          # s=6
    ]
    v_entries = [
        [(0,0,1), (1,1,1)],           # s=0
        [(0,0,1)],                     # s=1
        [(0,1,1), (1,1,-1)],          # s=2
        [(1,0,1), (0,0,-1)],          # s=3
        [(1,1,1)],                     # s=4
        [(0,0,1), (0,1,1)],           # s=5
        [(1,0,1), (1,1,1)],           # s=6
    ]

    def make_form(params, is_A):
        """Make a 2x2 list of linear forms based on params dict"""
        t = params['x'] if is_A else params['y']
        if t == 'single':
            br, bc = params['p']
            return single_block_form(br, bc)
        elif t == 'sum':
            br1, bc1 = params['p1']
            br2, bc2 = params['p2']
            return sum_block_form(br1, bc1, br2, bc2)
        elif t == 'diff':
            br1, bc1 = params['p1']
            br2, bc2 = params['p2']
            return diff_block_form(br1, bc1, br2, bc2)
        else:
            raise ValueError(f"Unknown form type: {t}")

    # Build U, V, W
    for k in range(7):
        x_params, x_side, y_params, y_side, c_contrib = top_products[k]
        X = make_form(x_params, True)   # 2x2 list of linear forms in A
        Y = make_form(y_params, False)  # 2x2 list of linear forms in B

        for s in range(7):
            r = 7 * k + s  # rank index

            # Build U linear form for this scalar multiplication
            # u is a sum of entries from X matrix
            u_form = {}  # {(global_i, global_j): coeff}
            for (lr, lc, coeff) in u_entries[s]:
                # X[lr][lc] is a linear form dict
                for (gi, gj), c in X[lr][lc].items():
                    u_form[(gi, gj)] = u_form.get((gi, gj), 0) + coeff * c

            # Build V linear form
            v_form = {}
            for (lr, lc, coeff) in v_entries[s]:
                for (gi, gj), c in Y[lr][lc].items():
                    v_form[(gi, gj)] = v_form.get((gi, gj), 0) + coeff * c

            # Fill U
            for (gi, gj), coeff in u_form.items():
                if coeff != 0:
                    U[gi * m + gj][r] += coeff

            # Fill V
            for (gi, gj), coeff in v_form.items():
                if coeff != 0:
                    V[gi * m + gj][r] += coeff

            # Fill W: the scalar product z_s contributes to P at positions D[s]
            # P is a 2x2 result. Each entry P[lr][lc] maps to a C block.
            # For each (block_r, block_c) in c_contrib with coefficient c_coeff:
            #   The block (block_r, block_c) of C gets contributions from P entries
            #   P[lr][lc] contributes to C entry (2*block_r+lr, 2*block_c+lc)
            for (br, bc), c_coeff in c_contrib.items():
                for lr in range(2):
                    for lc in range(2):
                        d_coeff = D[s][lr][lc]
                        if d_coeff == 0:
                            continue
                        total_coeff = c_coeff * d_coeff
                        if total_coeff == 0:
                            continue
                        # C global index: row = 2*br + lr, col = 2*bc + lc
                        i_global = 2 * br + lr
                        j_global = 2 * bc + lc
                        # W index: j * n + i (trace convention W[k*n+i, r])
                        W[j_global * n + i_global][r] += total_coeff

    return {"U": U, "V": V, "W": W}
