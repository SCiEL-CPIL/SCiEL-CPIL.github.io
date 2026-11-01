import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, seed=12345):
    """Exact verification of a tensor decomposition.

    Convention: C[i,k] = sum_j A[i,j] B[j,k], and
    C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]) W[k*n+i,c].
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(seed)
    for _ in range(6):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        Cc = np.zeros((n, p))
        for c in range(U.shape[1]):
            u = np.dot(U[:, c], A)
            v = np.dot(B, V[:, c])
            Cc += np.outer(u, v) * W[:, c]
        if not np.allclose(Cc, C, atol=1e-7):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin well under the 150s wall-clock kill

    n, m, p = DIMENSIONS

    def schoolbook():
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
        return U, V, W

    U, V, W = schoolbook()
    if not _verify(U, V, W, n, m, p):
        # schoolbook is mathematically exact; if verification ever fails, keep it anyway
        return {"U": U, "V": V, "W": W}

    # ---- Block decomposition using Strassen (rank 7) for 2x2*2x2 blocks ----
    def build_block():
        rank = 57
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(uo, vo, wo):
            nonlocal col
            # M1 = (a+d)(x+u)
            U[uo][col] += 1; U[uo+3][col] += 1
            V[vo][col] += 1; V[vo+2][col] += 1
            W[wo][col] += 1; W[wo+3][col] += 1; col += 1
            # M2 = (c+d)x
            U[uo+2][col] += 1; U[uo+3][col] += 1
            V[vo][col] += 1
            W[wo][col] -= 1; W[wo+2][col] += 1; col += 1
            # M3 = a(y-u)
            U[uo][col] += 1
            V[vo+1][col] += 1; V[vo+2][col] -= 1
            W[wo][col] += 1; W[wo+1][col] += 1; col += 1
            # M4 = d(v-u)
            U[uo+3][col] += 1
            V[vo+3][col] += 1; V[vo+2][col] -= 1
            W[wo][col] -= 1; W[wo+1][col] += 1; col += 1
            # M5 = (a+b)u
            U[uo][col] += 1; U[uo+1][col] += 1
            V[vo+2][col] += 1
            W[wo+2][col] += 1; W[wo+3][col] -= 1; col += 1
            # M6 = (c+d)(v-y)
            U[uo+2][col] += 1; U[uo+3][col] += 1
            V[vo+3][col] += 1; V[vo+1][col] -= 1
            W[wo+2][col] += 1; W[wo+3][col] += 1; col += 1
            # M7 = (a+b)(y-v)
            U[uo][col] += 1; U[uo+1][col] += 1
            V[vo+1][col] += 1; V[vo+3][col] -= 1
            W[wo+1][col] += 1; W[wo+3][col] += 1; col += 1

        def add_schoolbook_2x3x2(uo, vo, wo):
            nonlocal col
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        U[uo + i*3 + j][col] += 1
                        V[vo + j*2 + k][col] += 1
                        W[wo + i*2 + k][col] += 1
                        col += 1

        add_strassen_block(0, 0, 0)      # A1*B11  (2x2*2x2)
        add_strassen_block(0, 10, 4)     # A1*B12a (2x2*2x2)
        add_strassen_block(0, 14, 8)     # A1*B12b (2x2*2x2)
        add_schoolbook_2x3x2(4, 12, 0)   # A2*B21  (2x3*3x2)
        add_schoolbook_2x3x2(4, 18, 4)   # A2*B22a (2x3*3x2)
        add_schoolbook_2x3x2(4, 24, 8)   # A2*B22b (2x3*3x2)
        return U, V, W

    def strip_zero_columns(U, V, W):
        r = U[0][0] and len(U[0])
        r = len(U[0])
        keep = []
        for c in range(r):
            if any(U[i][c] != 0 for i in range(len(U))) or \
               any(V[i][c] != 0 for i in range(len(V))) or \
               any(W[i][c] != 0 for i in range(len(W))):
                keep.append(c)
        if len(keep) == r:
            return U, V, W
        U = [[row[c] for c in keep] for row in U]
        V = [[row[c] for c in keep] for row in V]
        W = [[row[c] for c in keep] for row in W]
        return U, V, W

    # ---- Refinement: greedy removal of linearly-redundant columns ----
    def refine(U, V, W, deadline):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        for _ in range(12):  # allow up to 12 removals
            if time.time() > deadline - 25.0:
                break
            if r <= 1:
                break
            removed = False
            for c in range(r):
                if time.time() > deadline - 25.0:
                    return U, V, W
                mask = [i for i in range(r) if i != c]
                Uc, Vc, Wc = U[:, mask], V[:, mask], W[:, mask]
                if _verify(Uc, Vc, Wc, n, m, p):
                    U, V, W = Uc, Vc, Wc
                    r -= 1
                    removed = True
                    break
            if not removed:
                break
        return U, V, W

    # Start from the stronger block construction if it verifies, else schoolbook
    U0, V0, W0 = build_block()
    U0, V0, W0 = strip_zero_columns(U0, V0, W0)
    if _verify(U0, V0, W0, n, m, p):
        U, V, W = U0, V0, W0
    else:
        U, V, W = schoolbook()

    # Refine (only if we have spare time)
    if time.time() < deadline - 20.0:
        U, V, W = refine(U, V, W, deadline)

    # Final safety: ensure a valid decomposition is returned
    if not _verify(U, V, W, n, m, p):
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
