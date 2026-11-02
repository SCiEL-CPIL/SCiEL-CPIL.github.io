import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8

    # Build the (2,2,8) matrix-multiplication tensor under the W[k*n+i, r] convention.
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def tensor_from(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W)

    def verify(U, V, W, tol=1e-9):
        return np.allclose(tensor_from(U, V, W), T, atol=tol, rtol=0.0)

    # Correct Strassen-based rank-28 decomposition: 4 independent 2x2x2 Strassen blocks.
    def strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # (A-part dict, B-part dict, C-part dict) for the 7 Strassen terms.
        # A part: (i,j)  -> coeff for A[i,j]
        # B part: (j,k)  -> coeff for B[j,k]
        # C part: (i,k)  -> coeff added to C[i,k]
        terms = [
            ({(0, 0): 1, (1, 1): 1},  {(0, 0): 1, (1, 1): 1},  {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1},  {(0, 0): 1},              {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1},              {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1},              {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1},   {(1, 1): 1},              {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1},  {(0, 0): 1, (0, 1): 1},   {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1},  {(1, 0): 1, (1, 1): 1},   {(0, 0): 1}),
        ]

        for q in range(p // 2):
            k0 = 2 * q
            for r, (A, B, C) in enumerate(terms):
                t = q * 7 + r
                for (i, j), c in A.items():
                    U[i * m + j, t] = c
                for (j, k), c in B.items():
                    V[j * p + (k0 + k), t] = c
                for (i, k), c in C.items():
                    W[(k0 + k) * n + i, t] = c
        return U, V, W

    start = time.time()
    deadline = 145.0

    U28, V28, W28 = strassen_28()
    ok28 = verify(U28, V28, W28)
    if not ok28:
        # Fallback: schoolbook rank-32 (always valid).
        U32 = np.zeros((n * m, n * m * p), dtype=float)
        V32 = np.zeros((m * p, n * m * p), dtype=float)
        W32 = np.zeros((p * n, n * m * p), dtype=float)
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U32[i * m + j, t] = 1.0
                    V32[j * p + k, t] = 1.0
                    W32[k * n + i, t] = 1.0
                    t += 1
        U28, V28, W28, ok28 = U32, V32, W32, True

    best_U, best_V, best_W = U28, V28, W28
    best_rank = U28.shape[1]

    # Exact-rational (±1) search for lower rank. Bounded by time.
    def try_low_rank(rank, seed, max_samples):
        rng = np.random.default_rng(seed)
        # Number of ±1 patterns per factor.
        nU = n * m          # 4
        nV = m * p          # 16
        nW = p * n          # 16
        for _ in range(max_samples):
            if time.time() > deadline - 10:
                break
            U = rng.choice([-1.0, 1.0], size=(nU, rank))
            V = rng.choice([-1.0, 1.0], size=(nV, rank))
            W = rng.choice([-1.0, 1.0], size=(nW, rank))
            if verify(U, V, W, tol=1e-9):
                return U, V, W
        return None

    for target in [27, 26, 25]:
        if time.time() > deadline - 30:
            break
        found = try_low_rank(target, seed=1000 + target, max_samples=4000)
        if found is not None:
            U, V, W = found
            if verify(U, V, W, tol=1e-9):
                best_U, best_V, best_W = U, V, W
                best_rank = target
                break

    if time.time() > deadline - 3:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
