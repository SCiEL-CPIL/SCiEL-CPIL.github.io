import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140

    nm, mp, pn = n * m, m * p, p * n

    # Build tensor T (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Lower bound from flattening ranks
    r1 = np.linalg.matrix_rank(T.reshape(nm, mp*pn))
    r2 = np.linalg.matrix_rank(T.transpose(1,0,2).reshape(mp, nm*pn))
    r3 = np.linalg.matrix_rank(T.transpose(2,0,1).reshape(pn, nm*mp))
    lb = int(max(r1, r2, r3))

    def verify_half(U, V, W):
        # U, V, W are half-integers (multiples of 0.5)
        # Scale by 2 to get integers
        Ui = np.round(2.0 * U).astype(np.int64)
        Vi = np.round(2.0 * V).astype(np.int64)
        Wi = np.round(2.0 * W).astype(np.int64)
        # Check that they're actually half-integers
        if not np.allclose(U, Ui / 2.0) or not np.allclose(V, Vi / 2.0) or not np.allclose(W, Wi / 2.0):
            return False
        # For each (i,j,k): sum_c Ui[a,c]*Vi[b,c]*Wi[c,c] should be 8 (since 2^3 * 1 = 8)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    s = int(np.sum(Ui[a, :] * Vi[b, :] * Wi[c, :]))
                    if s != 8:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    U0, V0, W0, rank0 = schoolbook()
    best = (U0, V0, W0, rank0)

    def als(U, V, W, r, max_iter=300):
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    a = i*m+j
                    g = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(r)] for k in range(p)])
                    sol, *_ = np.linalg.lstsq(g, np.ones(p), rcond=None)
                    U[a, :] = sol
            # Update V
            for j in range(m):
                for k in range(p):
                    b = j*p+k
                    h = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(r)] for i in range(n)])
                    sol, *_ = np.linalg.lstsq(h, np.ones(n), rcond=None)
                    V[b, :] = sol
            # Update W
            for k in range(p):
                for i in range(n):
                    c = k*n+i
                    f = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                    sol, *_ = np.linalg.lstsq(f, np.ones(m), rcond=None)
                    W[c, :] = sol
        return U, V, W

    rng = np.random.RandomState(42)
    for r in range(lb, rank0):
        if time.time() > deadline - 15:
            break
        found = False
        for restart in range(200):
            if time.time() > deadline - 15:
                break
            # Initialize with random half-integers
            U = rng.randint(-4, 5, size=(nm, r)) / 2.0
            V = rng.randint(-4, 5, size=(mp, r)) / 2.0
            W = rng.randint(-4, 5, size=(pn, r)) / 2.0
            U, V, W = als(U, V, W, r)
            # Round to half-integers
            Ur = np.round(2.0 * U) / 2.0
            Vr = np.round(2.0 * V) / 2.0
            Wr = np.round(2.0 * W) / 2.0
            if verify_half(Ur, Vr, Wr):
                found = True
                best = (Ur, Vr, Wr, r)
                break
        if found:
            break

    U, V, W, rank = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
