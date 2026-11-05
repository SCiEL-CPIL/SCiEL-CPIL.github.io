import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)

    # Part 1: 4x4x4 multiplication using rank-49 nested Strassen decomposition
    # We split A, B, C into 2x2 blocks of 2x2 matrices.
    # Outer level: 7 Strassen products of 2x2 blocks
    # Inner level: each 2x2x2 product uses 7 Strassen scalar products
    # Total: 7 * 7 = 49 terms

    # Define the 7 Strassen products for 2x2x2 (inner level)
    # Each product: (u_terms, v_terms, w_terms)
    # u_terms: list of (i, k, coeff) for 2x2 A
    # v_terms: list of (k, j, coeff) for 2x2 B
    # w_terms: list of (j, i, coeff) for 2x2 C (using W[k*n+i] convention, so w index is j*n+i)

    inner_strassen = [
        # M1 = (a+d)(e+h) -> C00+1, C11+1
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        # M2 = (c+d)e -> C10+1, C11-1
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        # M3 = a(f-h) -> C01+1, C11+1
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        # M4 = d(g-e) -> C00+1, C10+1
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        # M5 = (a+b)h -> C00-1, C01+1
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        # M6 = (c-a)(e+f) -> C11+1
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        # M7 = (b-d)(g+h) -> C00+1
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]

    # Outer level: 7 Strassen products for 4x4x4
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)

    # Outer Strassen products:
    # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2 = (A10+A11)B00 -> C10+1, C11-1
    # M3 = A00(B01-B11) -> C01+1, C11+1
    # M4 = A11(B10-B00) -> C00+1, C10+1
    # M5 = (A00+A01)B11 -> C00-1, C01+1
    # M6 = (A10-A00)(B00+B01) -> C11+1
    # M7 = (A01-A11)(B10+B11) -> C00+1

    outer_strassen = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},
    ]

    def a_global(block, li, lk):
        """Map local 2x2 indices to global 4x4 indices for A."""
        if block == 0:  # A00
            return li, lk
        elif block == 1:  # A01
            return li, lk + 2
        elif block == 2:  # A10
            return li + 2, lk
        elif block == 3:  # A11
            return li + 2, lk + 2

    def b_global(block, lk, lj):
        """Map local 2x2 indices to global 4x4 indices for B."""
        if block == 0:  # B00
            return lk, lj
        elif block == 1:  # B01
            return lk, lj + 2
        elif block == 2:  # B10
            return lk + 2, lj
        elif block == 3:  # B11
            return lk + 2, lj + 2

    def c_global(block, li, lj):
        """Map local 2x2 indices to global 4x4 indices for C."""
        if block == 0:  # C00
            return li, lj
        elif block == 1:  # C01
            return li, lj + 2
        elif block == 2:  # C10
            return li + 2, lj
        elif block == 3:  # C11
            return li + 2, lj + 2

    # Generate the 49 terms for 4x4x4
    for prod_idx, prod in enumerate(outer_strassen):
        a_form = prod['A']
        b_form = prod['B']
        c_form = prod['C']

        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)  # 4*5 = 20 entries, but we only use first 16 for 4x4
            v_vec = [0.0] * (m * p)  # 5*5 = 25 entries, but we only use first 16 for 4x4
            w_vec = [0.0] * (p * n)  # 5*4 = 20 entries, but we only use first 16 for 4x4

            # Fill u_vec with contributions from A blocks
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign

            # Fill v_vec with contributions from B blocks
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign

            # Fill w_vec with contributions from C blocks
            # W index: j*n + i (trace convention W[k*n+i] where k=j, i=i)
            for c_block, c_sign in c_form.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign

            add_rank1(u_vec, v_vec, w_vec)

    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
    # A[i,k] * B[k,4] contributes to C[i,4]
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)

    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook (20 terms)
    # A[i,4] * B[4,j] contributes to C[i,j]
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)

    rank = len(U_cols)

    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
