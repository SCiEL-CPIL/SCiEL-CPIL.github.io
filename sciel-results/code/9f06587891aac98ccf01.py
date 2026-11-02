"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 108.0

    # Strategy: use 2x2x2 Strassen blocks (rank 7) where possible,
    # and 2x1x2 standard blocks (rank 2 per k) for remaining columns.
    #
    # Partition of m=5: {0,1}, {2,3}, {4}
    # Partition of p=6: {0,1}, {2,3}, {4,5}
    #
    # For each (j-block, k-block) pair:
    #   If j-block has size 2 and k-block has size 2: Strassen, rank 7
    #   If j-block has size 2 and k-block has size 1: standard, rank 4 (2 rows * 2 cols * 1... wait)
    #   Actually for 2x1x2: rank = 2*1*2 = 4
    #   For 2x2x1: rank = 2*2*1 = 4
    #
    # j-blocks: sizes [2, 2, 1]
    # k-blocks: sizes [2, 2, 2]
    #
    # (j=0,1) x (k=0,1): 2x2x2 -> 7
    # (j=0,1) x (k=2,3): 2x2x2 -> 7
    # (j=0,1) x (k=4,5): 2x2x2 -> 7
    # (j=2,3) x (k=0,1): 2x2x2 -> 7
    # (j=2,3) x (k=2,3): 2x2x2 -> 7
    # (j=2,3) x (k=4,5): 2x2x2 -> 7
    # (j=4)   x (k=0,1): 2x1x2 -> 4
    # (j=4)   x (k=2,3): 2x1x2 -> 4
    # (j=4)   x (k=4,5): 2x1x2 -> 4
    #
    # Total: 6*7 + 3*4 = 42 + 12 = 54

    R = 54
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    c = 0

    def put(u_pairs, v_pairs, w_pairs):
        nonlocal c
        for idx, val in u_pairs:
            U[idx, c] += val
        for idx, val in v_pairs:
            V[idx, c] += val
        for idx, val in w_pairs:
            W[idx, c] += val
        c += 1

    # Strassen 2x2x2 for rows i in {0,1}, cols j in {ja,jb}, out k in {ka,kb}
    def strassen(ja, jb, ka, kb):
        ua0 = 0 * m + ja
        ub0 = 0 * m + jb
        ua1 = 1 * m + ja
        ub1 = 1 * m + jb
        va_ka = ja * p + ka
        va_kb = ja * p + kb
        vb_ka = jb * p + ka
        vb_kb = jb * p + kb
        wk0_0 = ka * n + 0
        wk0_1 = ka * n + 1
        wk1_0 = kb * n + 0
        wk1_1 = kb * n + 1

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        put([(ua0, 1), (ub1, 1)], [(va_ka, 1), (vb_kb, 1)], [(wk0_0, 1), (wk1_1, 1)])
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        put([(ua1, 1), (ub1, 1)], [(va_ka, 1)], [(wk0_1, 1), (wk1_1, -1)])
        # M3 = A00(B10-B00) -> C00 -= M3, C01 += M3
        put([(ua0, 1)], [(va_kb, 1), (vb_kb, -1)], [(wk1_0, 1), (wk1_1, 1)])
        # M4 = A11(B01-B00) -> C00 += M4, C01 += M4
        put([(ub1, 1)], [(vb_ka, 1), (va_ka, -1)], [(wk0_0, 1), (wk0_1, 1)])
        # M5 = (A00+A01)B11 -> C00 -= M5, C10 += M5
        put([(ua0, 1), (ub0, 1)], [(vb_kb, 1)], [(wk0_0, -1), (wk1_0, 1)])
        # M6 = (A10-A00)(B00+B10) -> C11 += M6
        put([(ua1, 1), (ua0, -1)], [(va_ka, 1), (va_kb, 1)], [(wk1_1, 1)])
        # M7 = (A01-A11)(B01+B11) -> C00 += M7
        put([(ub0, 1), (ub1, -1)], [(vb_ka, 1), (vb_kb, 1)], [(wk0_0, 1)])

    # Six Strassen blocks: j in {0,1}x{2,3}, k in {0,1}x{2,3}x{4,5}
    for (ja, jb) in [(0, 1), (2, 3)]:
        for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
            strassen(ja, jb, ka, kb)

    # Three 2x1x2 standard blocks for j=4, k in {0,1},{2,3},{4,5}
    for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
        for i in range(n):
            for k in [ka, kb]:
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

    if c != R:
        # Should not happen, but fallback
        R = c

    if time.time() > deadline:
        return {"U": U[:, :R].tolist(), "V": V[:, :R].tolist(), "W": W[:, :R].tolist()}

    # Vectorized verification
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j, :R] * V[j * p + k, :R], W[k * n + i, :R]))
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: trivial rank 60
        Uf = np.zeros((n * m, n * m * p), dtype=float)
        Vf = np.zeros((m * p, n * m * p), dtype=float)
        Wf = np.zeros((p * n, n * m * p), dtype=float)
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, cc] = 1.0
                    Vf[j * p + k, cc] = 1.0
                    Wf[k * n + i, cc] = 1.0
                    cc += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": U[:, :R].tolist(), "V": V[:, :R].tolist(), "W": W[:, :R].tolist()}
# EVOLVE-BLOCK-END
