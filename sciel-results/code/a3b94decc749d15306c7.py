"""Matrix multiplication 2x3x7 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2,3,7 : A is n x m (2x3), B is m x p (3x7), C is n x p (2x7)
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def to_lists(U, V, W):
        return {"U": [[int(round(x)) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in U],
                "V": [[int(round(x)) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in V],
                "W": [[int(round(x)) if abs(x - round(x)) < 1e-6 else float(x) for x in row] for row in W]}

    def valid(U, V, W, r):
        # C[i,k] = sum_c U[i*m+j... ] ; verify on random inputs
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for _ in range(5):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_true = A @ B
            # Compute via decomposition: C[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k]) * W[i,k,c]
            # U indexed (i*m+j, c), V (j*p+k, c), W (k*n+i, c)
            C = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        ua = 0.0
                        for j in range(m):
                            ua += Ua[i*m+j, c] * A[i, j]
                        vb = 0.0
                        for j in range(m):
                            vb += Va[j*p+k, c] * B[j, k]
                        s += ua * vb * Wa[k*n+i, c]
                    C[i, k] = s
            if not np.allclose(C, C_true, atol=1e-6):
                return False
        return True

    # Try known optimal rank 2p + 2n - 2 = 18 via random search
    target = 2*p + 2*n - 2
    rng = np.random.default_rng(0)
    for r in [target, target+1, target+2]:
        if time.time() > deadline - 10:
            break
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            U = rng.integers(-2, 3, size=(n*m, r))
            V = rng.integers(-2, 3, size=(m*p, r))
            W = rng.integers(-2, 3, size=(p*n, r))
            if valid(U.tolist(), V.tolist(), W.tolist(), r):
                best = to_lists(U.tolist(), V.tolist(), W.tolist())
                return best
    return best
