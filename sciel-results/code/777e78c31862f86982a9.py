import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback (Rank 48)
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
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Build a rank-32 decomposition using 4 blocks of rank-8.
    # Each block covers 2 output columns (k0, k0+1) and uses:
    #   - 7 Strassen terms for the 2x2x2 sub-tensor (rows i in {0,1}, cols a in {0,1})
    #   - 1 outer product term for the remaining row a=2
    # Total rank = 4 * 8 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 rank-7 decomposition
    # Indices: (i, a) for U, (a, k) for V, (k, i) for W
    # where i, a, k are local indices in {0, 1}
    
    # S_U[r][i][a], S_V[r][a][k], S_W[r][k][i]
    S_U = [
        [[1, 0], [0, 1]],   # M1: (A00+A11)
        [[0, 0], [1, 1]],   # M2: (A10+A11)
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: (A00+A01)
        [[1, 0], [1, -1]],  # M6: (A10-A00)
        [[0, 1], [0, -1]],  # M7: (A01-A11)
    ]
    S_V = [
        [[1, 0], [0, 1]],   # (B00+B11)
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # (B01-B11)
        [[0, 1], [1, 0]],   # (B10-B00)
        [[0, 0], [0, 1]],   # B11
        [[1, 1], [0, 0]],   # (B00+B01)
        [[0, 0], [1, 1]],   # (B10+B11)
    ]
    S_W = [
        [[1, 0], [0, 1]],   # (C00+C11)
        [[0, 1], [0, -1]],  # (C01-C11)
        [[0, 1], [0, 1]],   # (C01+C11)
        [[1, 1], [0, 0]],   # (C00+C01)
        [[-1, 1], [0, 0]],  # (C01-C00)
        [[0, 0], [0, 1]],   # C11
        [[1, 0], [0, 0]],   # C00
    ]

    nblocks = p // 2  # 4 blocks

    for b in range(nblocks):
        k0 = 2 * b
        base_col = b * 8

        # 7 Strassen terms
        for r in range(7):
            col = base_col + r
            # Fill U: rows are i*m + a, for i in {0,1}, a in {0,1}
            for i in range(2):
                for a in range(2):
                    val = S_U[r][i][a]
                    if val != 0:
                        U[i * m + a][col] += val
            
            # Fill V: rows are a*p + k, for a in {0,1}, k in {k0, k0+1}
            for a in range(2):
                for dk in range(2):
                    k_local = dk
                    val = S_V[r][a][k_local]
                    if val != 0:
                        V[a * p + (k0 + k_local)][col] += val
            
            # Fill W: rows are k*n + i, for k in {k0, k0+1}, i in {0,1}
            for dk in range(2):
                for i in range(2):
                    val = S_W[r][dk][i]
                    if val != 0:
                        W[(k0 + dk) * n + i][col] += val

            if not _elapsed_ok():
                return _schoolbook()

        # 1 outer product term for a=2
        col = base_col + 7
        for i in range(2):
            for dk in range(2):
                k = k0 + dk
                U[i * m + 2][col] += 1
                V[2 * p + k][col] += 1
                W[k * n + i][col] += 1

        if not _elapsed_ok():
            return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()
# EVOLVE-BLOCK-END
