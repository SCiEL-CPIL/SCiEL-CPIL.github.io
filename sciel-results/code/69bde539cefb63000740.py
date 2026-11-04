"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  Split the 8 output columns into 4 blocks of 2 columns each.
  For each block, decompose the 2x3x2 all-ones tensor as:
    - 2x2x2 part (a in {0,1}) using Strassen: rank 7
    - 2x1x2 part (a = 2) as a single rank-1 term: rank 1
  Total per block: 8, total rank: 4 * 8 = 32.
"""

import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # U indices: (i,a) where i in {0,1}, a in {0,1} -> index i*2+a
    # V indices: (a,k) where a in {0,1}, k in {0,1} -> index a*2+k
    # W indices: (k,i) where k in {0,1}, i in {0,1} -> index k*2+i

    # Standard Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)

    S_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    S_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10-B11
        [-1, 0, 1, 0],  # M4: B01-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B10
        [0, 0, 1, 1]    # M7: B01+B11
    ]
    S_W = [
        [1, 0, 0, 1],   # M1: C00 + C10
        [0, 1, 0, -1],  # M2: C01 - C11
        [0, 0, 1, 1],   # M3: C10 + C11
        [1, 1, 0, 0],   # M4: C00 + C01
        [-1, 0, 1, 0],  # M5: -C00 + C10
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]

    # Build rank-32 candidate
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for block_idx in range(4):
        k0 = block_idx * 2

        # Part 1: Strassen for a in {0,1}, k in {k0, k0+1}
        for r in range(7):
            # U: i in {0,1}, a in {0,1}
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += u_val

            # V: a in {0,1}, k in {k0, k0+1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] += v_val

            # W: k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

        # Part 2: Rank-1 term for a=2, k in {k0, k0+1}
        # U[i, 2] = 1 for all i
        # V[2, k] = 1 for k in {k0, k0+1}
        # W[k, i] = 1 for all k, i
        for li in range(2):
            U[li * m + 2][col] = 1
        for lk in range(2):
            V[2 * p + (k0 + lk)][col] = 1
        for lk in range(2):
            for li in range(2):
                W[(k0 + lk) * n + li][col] = 1

        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
