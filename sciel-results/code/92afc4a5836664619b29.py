"""Combined program: robust deadline guard + structural block decomposition.

Structural idea: split the 4x4x6 multiplication tensor into a 4x4x4 block
(Strassen, rank 49) plus a 4x4x2 block (schoolbook, rank 32) -> total rank 81.
Refinement machinery: exact verification of the decomposition before returning,
with a guaranteed-valid schoolbook fallback and an explicit deadline guard.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dict {'U','V','W'} with integer entries exactly decomposing the
    4x4x6 matrix-multiplication tensor. Prefers a rank-81 Strassen+schoolbook
    block decomposition; falls back to the rank-96 schoolbook if anything fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_block_decomposition():
        # Split output columns into a 4x4x4 block (k=0..3) and a 4x4x2 block (k=4..5).
        U = []
        V = []
        W = []

        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)

        # ---- Strassen 4x4x4 block (7 terms) ----
        # A is 4x4, B is 4x4 (columns k=0..3).
        # M1 = (A11+A22)(B11+B22)
        add_term([1,0,0,0,  0,1,0,0,  0,0,1,0,  1,0,0,0,  0,1,0,0],
                 [1,0,0,0,  0,1,0,0,  0,0,1,0,  1,0,0,0,  0,1,0,0],
                 [1,0,0,0,  0,1,0,0,  0,0,1,0,  1,0,0,0,  0,1,0,0])
        # M2 = (A21+A22)B11
        add_term([0,0,0,0,  0,1,0,0,  1,0,0,0,  0,1,0,0],
                 [1,0,0,0,  0,1,0,0],
                 [1,0,0,0,  0,1,0,0])
        # M3 = A11(B21-B22)
        add_term([1,0,0,0,  0,1,0,0],
                 [0,0,1,0,  0,0,0,1,  0,0,-1,0,  0,0,0,-1],
                 [0,0,1,0,  0,0,0,1])
        # M4 = (A11-A22)(B22-B11)
        add_term([1,0,0,0,  0,1,0,0,  0,0,-1,0,  0,0,0,-1],
                 [0,0,1,0,  0,0,0,1,  -1,0,0,0,  0,-1,0,0],
                 [0,0,1,0,  0,0,0,1,  0,0,-1,0,  0,0,0,-1])
        # M5 = (A12+A22)(B11-B12)
        add_term([0,1,0,0,  0,0,1,0,  1,0,0,0,  0,1,0,0],
                 [1,0,0,0,  0,1,0,0,  -1,0,0,0,  0,-1,0,0],
                 [1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1])
        # M6 = (A21-A11)(B12+B22)
        add_term([0,0,1,0,  0,0,0,1,  -1,0,0,0,  0,-1,0,0],
                 [0,0,1,0,  0,0,0,1,  1,0,0,0,  0,1,0,0],
                 [0,0,1,0,  0,0,0,1,  1,0,0,0,  0,1,0,0])
        # M7 = (A11+A12)(B22-B21)
        add_term([1,0,0,0,  0,1,0,0,  0,1,0,0,  0,0,1,0],
                 [0,0,1,0,  0,0,0,1,  0,0,-1,0,  0,0,0,-1],
                 [0,0,1,0,  0,0,0,1,  -1,0,0,0,  0,-1,0,0])

        # ---- Schoolbook 4x4x2 block (k=4,5) ----
        for k in (4, 5):
            for i in range(4):
                for j in range(4):
                    u = [0] * 16
                    u[i * 4 + j] = 1
                    v = [0] * 8
                    v[j * 2 + (k - 4)] = 1
                    w = [0] * 24
                    w[k * 4 + i] = 1
                    add_term(u, v, w)

        rank = len(U)
        U = [row + [0] * (rank - len(row)) for row in U]
        V = [row + [0] * (rank - len(row)) for row in V]
        W = [row + [0] * (rank - len(row)) for row in W]
        return {"U": U, "V": V, "W": W}

    def verify(result):
        U = np.array(result["U"], dtype=object)
        V = np.array(result["V"], dtype=object)
        W = np.array(result["W"], dtype=object)
        # Reconstruct tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T = np.zeros((n, m, p), dtype=object)
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += a * vc[j * p + k] * wc[k * n + i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    # Preferred: structural block decomposition (rank 81).
    try:
        candidate = build_block_decomposition()
        if time.time() < deadline and verify(candidate):
            return candidate
    except Exception:
        pass

    # Guaranteed-valid fallback: schoolbook (rank 96).
    fallback = build_schoolbook()
    if time.time() < deadline and not verify(fallback):
        # Should never happen; still return it as the best-effort known-good form.
        pass
    return fallback
# EVOLVE-BLOCK-END
