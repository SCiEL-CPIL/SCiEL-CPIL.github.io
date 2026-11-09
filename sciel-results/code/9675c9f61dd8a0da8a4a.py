import numpy as np
import time

_DIM = (6, 6, 6)

def construct():
    n, m, p = _DIM
    B = 2
    N_b = n // B
    rank = N_b * N_b * N_b * 7

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 2x2x2 terms
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (local_row, local_col, coeff)
    # V_entries: list of (local_row, local_col, coeff)
    # W_entries: list of (local_row, local_col, coeff)

    terms = [
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
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    ]

    # Iterate over all block triples
    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for t_idx in range(7):
                    col = (i * N_b + k) * (N_b * 7) + l * 7 + t_idx
                    u_entries, v_entries, w_entries = terms[t_idx]

                    for (lr, lc, c) in u_entries:
                        gr = i * B + lr
                        gc = k * B + lc
                        U[gr * m + gc, col] += c

                    for (lr, lc, c) in v_entries:
                        gr = k * B + lr
                        gc = l * B + lc
                        V[gr * p + gc, col] += c

                    for (lr, lc, c) in w_entries:
                        gr = i * B + lr
                        gc = l * B + lc
                        # W convention: W[k*n + i, r] where i is row of C, k is col of C
                        # So row index = gc * n + gr
                        W[gc * n + gr, col] += c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
