"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact decomposition of the 2x2 (x) 2x4 matrix-multiplication tensor.
    #
    # Structural argument:
    #   The 2x4 right factor splits into two independent 2x2 blocks (columns 0,1 and 2,3).
    #   The multiplication tensor is therefore the direct sum of two 2x2x2 multiplication
    #   tensors, so its rank equals rank(2x2x2) + rank(2x2x2) = 7 + 7 = 14.
    #   14 is a hard lower bound (each 2x2x2 block needs rank >= 7, the Strassen bound),
    #   so 14 is exactly minimal.
    #
    # We build it explicitly from the 7-term Strassen identity, placing the two copies in
    # disjoint column ranges [0,7) and [7,14).

    n, m, p = DIMENSIONS  # n=2, m=2, p=4

    # Strassen 7-term decomposition for 2x2 x 2x2 -> 2x2.
    # Conventions:
    #   A = [[a, b], [c, d]]   (2x2 left)
    #   B = [[e, f], [g, h]]   (2x2 right)
    #   C = A @ B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d) e
    #   M3 = a (f-h)
    #   M4 = d (g-e)
    #   M5 = (a+b) g
    #   M6 = (c-a) (e+f)
    #   M7 = (b-d) (h+g)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each factor is a linear form in the input entries, with integer coefficients.
    #
    # U index = i*m + j  (row i, col j of A)   ->  U[A-index, t]
    # V index = j*p + k  (row j, col k of B)   ->  V[B-index, t]
    # W index = k*n + i  (row k, col i of C)   ->  W[C-index, t]
    #
    # For a 2x2 block, A-index = i*2+j, B-index = j*2+k, C-index = k*2+i.

    # For each Strassen term t, list of (factor_index, coefficient) per factor.
    # A-indices: a=0, b=1, c=2, d=3
    # B-indices: e=0, f=1, g=2, h=3
    # C-indices: C00=0, C01=1, C10=2, C11=3

    strassen = [
        # t, U (A), V (B), W (C)
        (0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        (1, [(2, 1), (3, 1)], [(0, 1)],                             [(2, 1)]),  # M2
        (2, [(0, 1)],                               [(1, 1), (3, -1)], [(1, 1)]),  # M3
        (3, [(3, 1)],                               [(2, 1), (0, -1)], [(2, 1)]),  # M4
        (4, [(0, 1), (1, 1)], [(2, 1)],                             [(0, -1), (1, 1)]),  # M5
        (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)],                    [(3, 1)]),  # M6
        (6, [(1, 1), (3, -1)], [(3, 1), (2, 1)],                    [(0, 1), (1, 1)]),  # M7
    ]

    # Two independent 2x2 blocks over the 2x4 right factor:
    #   block 0 uses B-columns k in {0, 1}  (B-index = j*4 + k)
    #   block 1 uses B-columns k in {2, 3}  (B-index = j*4 + k)
    # C-columns likewise: C-index = k*2 + i with k in the block's column set.

    def build_block(col_offset):
        """Return (U, V, W) as dicts of {index: {t: coeff}} for one 2x2 block.
        col_offset is the B/C column offset (0 or 2)."""
        U = {}
        V = {}
        W = {}
        for t, u_terms, v_terms, w_terms in strassen:
            for (ai, coef) in u_terms:
                U.setdefault(ai, {})[t] = U.get(ai, {}).get(t, 0) + coef
            for (bj, coef) in v_terms:
                # B-index = j*4 + (k0 + bj_local) where bj_local in {0,1} maps to k=bj_local
                # Here bj is the local 2x2 B column index (0 or 1); global k = col_offset + bj
                b_global = bj  # local col; we add offset below
                # j is the row: we need to recover j from bj? bj is column in {0,1}.
                # We need the row j as well. Re-derive: v_terms entries are (col_local, coef)
                # but Strassen V forms also depend on row j. We must encode full B-index.
                pass
        return U, V, W

    # The above helper is getting tangled; build directly with explicit global indices.
    # A-index = i*2 + j.  For B, global index = j*4 + k.  For C, global index = k*2 + i.
    #
    # We define each Strassen term's factors in terms of (i, j, k) and coefficient,
    # then map to global indices per block.

    # Term definitions: (t, list of (i, j, coef) for A), (list of (j, k_local, coef) for B), (list of (k_local, i, coef) for C)
    # k_local in {0,1} is the local column within the 2x2 block.
    terms = [
        # M1 = (a+d)(e+h): A: (0,0),(1,1); B: (0,0),(1,1); C: (0,0),(1,1)
        (0, [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (c+d) e: A: (1,0),(1,1); B: (0,0); C: (0,1)
        (1, [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1)]),
        # M3 = a (f-h): A: (0,0); B: (0,1),(1,1); C: (1,0)
        (2, [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1)]),
        # M4 = d (g-e): A: (1,1); B: (1,0),(0,0); C: (0,1)
        (3, [(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 1, 1)]),
        # M5 = (a+b) g: A: (0,0),(0,1); B: (1,0); C: (0,0),(1,0)
        (4, [(0, 0, 1), (0, 1, 1)], [(1, 0, 1)], [(0, 0, -1), (1, 0, 1)]),
        # M6 = (c-a)(e+f): A: (1,0),(0,0); B: (0,0),(0,1); C: (1,1)
        (5, [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M7 = (b-d)(h+g): A: (0,1),(1,1); B: (1,1),(1,0); C: (0,0),(1,0)
        (6, [(0, 1, 1), (1, 1, -1)], [(1, 1, 1), (1, 0, 1)], [(0, 0, 1), (1, 0, 1)]),
    ]

    rank = 14
    U = [[0] * rank for _ in range(n * m)]   # 4 rows
    V = [[0] * rank for _ in range(m * p)]   # 8 rows
    W = [[0] * rank for _ in range(p * n)]   # 8 rows

    for block, col_offset in enumerate([0, 2]):
        base = block * 7  # column offset in the 14-column space
        for (t, a_terms, b_terms, c_terms) in terms:
            col = base + t
            for (i, j, coef) in a_terms:
                a_idx = i * m + j
                U[a_idx][col] += coef
            for (j, k_local, coef) in b_terms:
                k_global = col_offset + k_local
                b_idx = j * p + k_global
                V[b_idx][col] += coef
            for (k_local, i, coef) in c_terms:
                k_global = col_offset + k_local
                c_idx = k_global * n + i
                W[c_idx][col] += coef

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
