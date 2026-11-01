"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Split 5 into 4+1
    # 2x4x6: two 2x2x6 blocks, each with rank 21 (3 Strassen 2x2x2 blocks)
    # 2x1x6: rank 12 (standard)
    # Total: 2*21 + 12 = 54

    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen block for 2x2x2: (i∈{0,1}, j∈{ja,jb}, k∈{ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        ua = ja       # (0, ja)
        ub = 5 + ja   # (1, ja)
        va = jb       # (0, jb)
        vb = 5 + jb   # (1, jb)
        wka0 = 2 * ka   # (ka, 0)
        wka1 = 2 * ka + 1  # (ka, 1)
        wkb0 = 2 * kb   # (kb, 0)
        wkb1 = 2 * kb + 1  # (kb, 1)
        vka = ja * p + ka  # (ja, ka)
        vkb = ja * p + kb  # (ja, kb)
        vjb_ka = jb * p + ka  # (jb, ka)
        vjb_kb = jb * p + kb  # (jb, kb)

        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term([(ua, 1), (vb, 1)], [(vka, 1), (vjb_kb, 1)], [(wka0, 1), (wkb1, 1)])
        # M2: (A10+A11)(B00) -> C10 - C11
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wkb0, 1), (wkb1, -1)])
        # M3: A00(B01-B11) -> C01 + C11
        add_term([(ua, 1)], [(vkb, 1), (vjb_kb, -1)], [(wka1, 1), (wkb1, 1)])
        # M4: A11(B10-B00) -> -C00 + C10
        add_term([(vb, 1)], [(vjb_ka, 1), (vka, -1)], [(wka0, -1), (wkb0, 1)])
        # M5: (A00+A01)(B11) -> -C00 + C01
        add_term([(ua, 1), (va, 1)], [(vjb_kb, 1)], [(wka0, -1), (wka1, 1)])
        # M6: (A10-A00)(B00+B01) -> C00
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wka0, 1)])
        # M7: (A01-A11)(B10+B11) -> -C00
        add_term([(va, 1), (vb, -1)], [(vjb_ka, 1), (vjb_kb, 1)], [(wka0, -1)])

    # 2x4x6: j in {0,1,2,3}, k in {0,...,5}
    # Block 1: j in {0,1}, k in {0,1}
    strassen_block(0, 1, 0, 1)
    # Block 2: j in {0,1}, k in {2,3}
    strassen_block(0, 1, 2, 3)
    # Block 3: j in {0,1}, k in {4,5}
    strassen_block(0, 1, 4, 5)
    # Block 4: j in {2,3}, k in {0,1}
    strassen_block(2, 3, 0, 1)
    # Block 5: j in {2,3}, k in {2,3}
    strassen_block(2, 3, 2, 3)
    # Block 6: j in {2,3}, k in {4,5}
    strassen_block(2, 3, 4, 5)

    # 2x1x6: j=4, k in {0,...,5}
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
