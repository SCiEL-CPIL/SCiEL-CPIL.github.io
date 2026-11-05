import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Block decomposition: 5 = 2 + 2 + 1
    blocks = [(0, 2), (2, 4), (4, 5)]

    terms = []  # Each term: (U_dict, V_dict, W_dict)

    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    def add_strassen_2x2(r1, c1, k1):
        """Add 7 Strassen terms for a 2x2x2 block at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)

        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        # M2 = (A10+A11)B00 -> C10 - C11
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        # M3 = A00(B01-B11) -> C01 + C11
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        # M4 = A11(B10-B00) -> C00 + C10
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        # M5 = (A00+A01)B11 -> -C00 + C01
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        # M6 = (A10-A00)(B00+B01) -> C11
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        # M7 = (A01-A11)(B10+B11) -> C00
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})

    def add_rank3_2x2x1(r1, c1, k1):
        """
        2x2x1 block: A is 2x2, B is 2x1, C is 2x1.
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        Rank-3 decomposition:
        M1 = (a00+a11)*b0 -> c0 += 1, c1 += 1
        M2 = (a10-a00)*b0 -> c0 += -1, c1 += 1
        M3 = (a01-a11)*b1 -> c0 += 1, c1 += -1
        """
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B0 = c1 * p + k1
        B1 = (c1 + 1) * p + k1

        C0 = k1 * n + r1
        C1 = k1 * n + (r1 + 1)

        add_term({A00: 1, A11: 1}, {B0: 1}, {C0: 1, C1: 1})
        add_term({A10: 1, A00: -1}, {B0: 1}, {C0: -1, C1: 1})
        add_term({A01: 1, A11: -1}, {B1: 1}, {C0: 1, C1: -1})

    def add_rank3_2x1x2(r1, c1, k1):
        """
        2x1x2 block: A is 2x1, B is 1x2, C is 2x2.
        c00 = a0*b0, c01 = a0*b1
        c10 = a1*b0, c11 = a1*b1
        Rank-3 decomposition:
        M1 = a0*(b0+b1) -> c00 += 1, c01 += 1
        M2 = a1*(b0+b1) -> c10 += 1, c11 += 1
        M3 = (a0-a1)*b1 -> c01 += 1, c11 += -1
        """
        A0 = r1 * m + c1
        A1 = (r1 + 1) * m + c1

        B0 = c1 * p + k1
        B1 = c1 * p + (k1 + 1)

        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)

        add_term({A0: 1}, {B0: 1, B1: 1}, {C00: 1, C01: 1})
        add_term({A1: 1}, {B0: 1, B1: 1}, {C10: 1, C11: 1})
        add_term({A0: 1, A1: -1}, {B1: 1}, {C01: 1, C11: -1})

    def add_rank3_1x2x2(r1, c1, k1):
        """
        1x2x2 block: A is 1x2, B is 2x2, C is 1x2.
        c0 = a0*b00 + a1*b10
        c1 = a0*b01 + a1*b11
        Rank-3 decomposition:
        M1 = a0*(b00+b01) -> c0 += 1, c1 += 1
        M2 = a1*(b10+b11) -> c0 += 1, c1 += 1
        M3 = (a0-a1)*(b11-b01) -> c0 += -1, c1 += 1
        """
        A0 = r1 * m + c1
        A1 = r1 * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C0 = k1 * n + r1
        C1 = (k1 + 1) * n + r1

        add_term({A0: 1}, {B00: 1, B01: 1}, {C0: 1, C1: 1})
        add_term({A1: 1}, {B10: 1, B11: 1}, {C0: 1, C1: 1})
        add_term({A0: 1, A1: -1}, {B11: 1, B01: -1}, {C0: -1, C1: 1})

    def add_schoolbook(r1, c1, k1, ar, ac, bc):
        """Add schoolbook terms for a block of size ar x ac x bc."""
        for ii in range(ar):
            for jj in range(ac):
                for kk in range(bc):
                    u_idx = (r1 + ii) * m + (c1 + jj)
                    v_idx = (c1 + jj) * p + (k1 + kk)
                    w_idx = (k1 + kk) * n + (r1 + ii)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Iterate over all block combinations
    for r_idx, (r1, r2) in enumerate(blocks):
        for c_idx, (c1, c2) in enumerate(blocks):
            for k_idx, (k1, k2) in enumerate(blocks):
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                elif ar == 2 and ac == 2 and bc == 1:
                    add_rank3_2x2x1(r1, c1, k1)
                elif ar == 2 and ac == 1 and bc == 2:
                    add_rank3_2x1x2(r1, c1, k1)
                elif ar == 1 and ac == 2 and bc == 2:
                    add_rank3_1x2x2(r1, c1, k1)
                else:
                    add_schoolbook(r1, c1, k1, ar, ac, bc)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    rank = len(terms)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
