import time

def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication via recursive Strassen.
    Uses a flat-index precomputation approach for speed and clarity.
    """
    deadline = time.time() + 140
    n = 4
    N = n * n
    rank = 49

    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]

    # Flat index for 2x2 block (br,bc) entry (r,s) in 4x4 matrix
    # A[i,j] -> i*4+j; block entry: i=2*br+r, j=2*bc+s
    def bidx(br, bc, r, s):
        return (2 * br + r) * 4 + (2 * bc + s)

    # Precompute the 7 Strassen block-level products
    # Each: X = linear combo of A-blocks, Y = linear combo of B-blocks, C = contributions to C-blocks
    # Block indices: (row_block, col_block) in {0,1}^2
    block_prods = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Precompute the 7 Strassen scalar-level products for 2x2
    # Each: X_entries [(r,s,coeff)], Y_entries [(r,s,coeff)], Z_contrib {(zr,zs):coeff}
    scalar_prods = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1,(1,1):1}),
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1,(1,1):-1}),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1,(1,1):1}),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1,(1,0):1}),
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1,(0,1):1}),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]

    # For each block product, precompute the flat-index forms of X and Y
    # X_form[(er,es)] = dict of flat_idx -> coeff
    # Y_form[(er,es)] = dict of flat_idx -> coeff
    block_forms = []
    for X_refs, Y_refs, C_refs in block_prods:
        X_form = {}
        Y_form = {}
        for (br, bc, coeff) in X_refs:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    X_form.setdefault((r, s), {})
                    X_form[(r, s)][idx] = X_form[(r, s)].get(idx, 0) + coeff
        for (br, bc, coeff) in Y_refs:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    Y_form.setdefault((r, s), {})
                    Y_form[(r, s)][idx] = Y_form[(r, s)].get(idx, 0) + coeff
        block_forms.append((X_form, Y_form, C_refs))

    # Build the 49 rank-1 terms
    r_idx = 0
    for X_form, Y_form, C_refs in block_forms:
        for X_entries, Y_entries, Z_contrib in scalar_prods:
            U_dict = {}
            V_dict = {}
            W_dict = {}

            for (er, es, ecoeff) in X_entries:
                if (er, es) in X_form:
                    for flat_idx, val in X_form[(er, es)].items():
                        U_dict[flat_idx] = U_dict.get(flat_idx, 0) + ecoeff * val

            for (er, es, ecoeff) in Y_entries:
                if (er, es) in Y_form:
                    for flat_idx, val in Y_form[(er, es)].items():
                        V_dict[flat_idx] = V_dict.get(flat_idx, 0) + ecoeff * val

            for (zr, zs), zcoeff in Z_contrib.items():
                for (cb_r, cb_c, ccoeff) in C_refs:
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zs
                    w_idx = col * n + row
                    W_dict[w_idx] = W_dict.get(w_idx, 0) + zcoeff * ccoeff

            for flat_idx, val in U_dict.items():
                if val != 0:
                    U[flat_idx][r_idx] = int(val)
            for flat_idx, val in V_dict.items():
                if val != 0:
                    V[flat_idx][r_idx] = int(val)
            for flat_idx, val in W_dict.items():
                if val != 0:
                    W[flat_idx][r_idx] = int(val)

            r_idx += 1
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    return {"U": U, "V": V, "W": W}
