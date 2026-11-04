"""Rewritten program using block-decomposition with Strassen for 2x2x2 sub-blocks."""

import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # We decompose the multiplication into:
    # - j=2 slice: 2x1x10, trivial rank 20
    # - j=0,1 slice: 2x2x10, split into 5 blocks of 2x2x2, each using Strassen (rank 7)
    # Total rank = 20 + 5*7 = 55

    R = 20 + 5 * 7
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)

    # Part 1: j=2 slice (trivial)
    # For each (i, k), one term
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            r += 1

    # Part 2: j=0,1 slice, split into 5 blocks of width 2
    # Strassen's 7 terms for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each block l (0..4), local k in {0,1}, global k = l*2 + k_local
    # Local i in {0,1}, local j in {0,1}
    # Global i = i, global j = j

    # Define Strassen coefficients
    # Each term has: A_coeffs[i_local][j_local], B_coeffs[j_local][k_local], C_coeffs[i_local][k_local]
    # We'll encode as: for term t, list of (i_local, j_local, coeff_A), (j_local, k_local, coeff_B), (i_local, k_local, coeff_C)

    strassen_terms = []

    # M1 = (a+d)(e+h): A: a=1, d=1; B: e=1, h=1; C: c00=1, c11=1
    strassen_terms.append([
        [(0,0,1), (1,1,1)],  # A coeffs
        [(0,0,1), (1,1,1)],  # B coeffs
        [(0,0,1), (1,1,1)],  # C coeffs
    ])

    # M2 = (c+d)e: A: c=1, d=1; B: e=1; C: c10=1
    strassen_terms.append([
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1)],
    ])

    # M3 = a(f-h): A: a=1; B: f=1, h=-1; C: c01=1
    strassen_terms.append([
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1)],
    ])

    # M4 = d(g-e): A: d=1; B: g=1, e=-1; C: c00=1, c10=1
    strassen_terms.append([
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (1,0,1)],
    ])

    # M5 = (a+b)h: A: a=1, b=1; B: h=1; C: c00=-1, c01=1
    strassen_terms.append([
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (0,1,1)],
    ])

    # M6 = (c-a)(e+f): A: c=1, a=-1; B: e=1, f=1; C: c11=1
    strassen_terms.append([
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
    ])

    # M7 = (b-d)(g+h): A: b=1, d=-1; B: g=1, h=1; C: c00=1
    strassen_terms.append([
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
    ])

    # Now fill in U, V, W for each block
    for l in range(5):
        for t in range(7):
            r = 20 + l * 7 + t
            A_coeffs, B_coeffs, C_coeffs = strassen_terms[t]

            for (i_loc, j_loc, ca) in A_coeffs:
                U[i_loc * m + j_loc, r] += ca

            for (j_loc, k_loc, cb) in B_coeffs:
                V[j_loc * p + (l * 2 + k_loc), r] += cb

            for (i_loc, k_loc, cc) in C_coeffs:
                W[(l * 2 + k_loc) * n + i_loc, r] += cc

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
