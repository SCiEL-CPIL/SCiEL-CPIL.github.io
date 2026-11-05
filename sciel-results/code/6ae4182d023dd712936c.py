import numpy as np
import time

n, m, p = 2, 4, 5

def verify_fast(U, V, W, n=2, m=4, p=5, tol=1e-8):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    U2 = U.reshape(n, m, rank)
    V2 = V.reshape(m, p, rank)
    W2 = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for rr in range(rank):
        T += U2[:,:,rr][:, :, None] * V2[:,:,rr][None, :, :] * W2[:,:,rr][None, None, :]
    return np.allclose(T, 1.0, atol=tol)

def try_rank(r, n=2, m=4, p=5, iters=200, seed=0, tol=1e-8):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n*m, r)) * 0.5
    V = rng.standard_normal((m*p, r)) * 0.5
    W = rng.standard_normal((p*n, r)) * 0.5
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    best_err = float('inf')
    for it in range(iters):
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V2[j, k, :] * W2[k, i, :]
                sol, *_ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U2[i, j, :] = sol
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U2[i, j, :] * W2[k, i, :]
                sol, *_ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                V2[j, k, :] = sol
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U2[i, j, :] * V2[j, k, :]
                sol, *_ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                W2[k, i, :] = sol
        T = np.zeros((n, m, p))
        for rr in range(r):
            T += U2[:,:,rr][:, :, None] * V2[:,:,rr][None, :, :] * W2[:,:,rr][None, None, :]
        err = np.linalg.norm(T - 1.0)
        if err < best_err:
            best_err = err
        if err < tol:
            break
    return best_err, U, V, W

for r in [30, 32, 34, 36, 40]:
    t0 = time.time()
    err, U, V, W = try_rank(r, iters=150, seed=0)
    dt = time.time() - t0
    valid = verify_fast(U, V, W, tol=1e-6)
    print(f"rank {r}: err={err:.6f}, valid={valid}, time={dt:.1f}s")
