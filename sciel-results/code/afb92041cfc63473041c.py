import time
import numpy as np

_DIM = (6, 6, 6)

def construct():
    n, m, p = _DIM
    B = 2
    N_b = n // B
    rank = N_b * N_b * N_b * 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 multiplication.
    # Each term is a list of (local_row, local_col, coeff).
    terms = (
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        ((0, 0, 1), (1, 1, 1),
         (0, 0, 1), (1, 1, 1),
         (0, 0, 1), (1, 1, 1)),
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        ((1, 0, 1), (1, 1, 1),
         (0, 0, 1),
         (1, 0, 1), (1, 1, -1)),
        # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
        ((0, 0, 1),
         (0, 1, 1), (1, 1, -1),
         (0, 1, 1), (1, 1, 1)),
        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        ((1, 1, 1),
         (1, 0, 1), (0, 0, -1),
         (0, 0, 1), (1, 0, 1)),
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        ((0, 0, 1), (0, 1, 1),
         (1, 1, 1),
         (0, 0, -1), (0, 1, 1)),
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        ((1, 0, 1), (0, 0, -1),
         (0, 0, 1), (0, 1, 1),
         (1, 1, 1)),
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        ((0, 1, 1), (1, 1, -1),
         (1, 0, 1), (1, 1, 1),
         (0, 0, 1)),
    )

    # Precompute per-block-pair term coefficients for speed.
    # For each (i,k,l) block triple, for each term t, we need:
    #   U entries: (u_row, coeff)
    #   V entries: (v_row, coeff)
    #   W entries: (w_row, coeff)
    # But we can just loop inline; 3*3*3*7 = 189 iterations, each small.

    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for t in range(7):
                    c = (i * N_b + k) * (N_b * 7) + l * 7 + t
                    term = terms[t]
                    # Parse: 3 groups of (lr, lc, coeff)
                    # Group 0: U, Group 1: V, Group 2: W
                    # Actually each group is a flat list of (lr, lc, coeff) tuples.
                    # Let's restructure: use separate lists for clarity.
                    pass

    # Rebuild with explicit separate lists for clarity and correctness.
    U_terms = (
        ((0,0,1),(1,1,1)),  # M1
        ((1,0,1),(1,1,1)),  # M2
        ((0,0,1)),           # M3
        ((1,1,1)),           # M4
        ((0,0,1),(0,1,1)),  # M5
        ((1,0,1),(0,0,-1)), # M6
        ((0,1,1),(1,1,-1)), # M7
    )
    V_terms = (
        ((0,0,1),(1,1,1)),  # M1
        ((0,0,1)),           # M2
        ((0,1,1),(1,1,-1)), # M3
        ((1,0,1),(0,0,-1)), # M4
        ((1,1,1)),           # M5
        ((0,0,1),(0,1,1)),  # M6
        ((1,0,1),(1,1,1)),  # M7
    )
    W_terms = (
        ((0,0,1),(1,1,1)),  # M1
        ((1,0,1),(1,1,-1)), # M2
        ((0,1,1),(1,1,1)),  # M3
        ((0,0,1),(1,0,1)),  # M4
        ((0,0,-1),(0,1,1)), # M5
        ((1,1,1)),           # M6
        ((0,0,1)),           # M7
    )

    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for t in range(7):
                    c = (i * N_b + k) * (N_b * 7) + l * 7 + t
                    for lr, lc, cf in U_terms[t]:
                        u_row = (i * B + lr) * m + (k * B + lc)
                        U[u_row][c] += cf
                    for lr, lc, cf in V_terms[t]:
                        v_row = (k * B + lr) * p + (l * B + lc)
                        V[v_row][c] += cf
                    for lr, lc, cf in W_terms[t]:
                        w_row = (l * B + lc) * n + (i * B + lr)
                        W[w_row][c] += cf

    return {"U": U, "V": V, "W": W}
