import numpy as np
import time
import sys

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130

    def schoolbook():
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    def compute_tensor_fast(U, V, W):
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for rr in range(rank):
            u = U[:, rr].reshape(n, m)
            v = V[:, rr].reshape(m, p)
            w = W[:, rr].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return T

    def verify_tensor(U, V, W, tol=1e-10):
        T = compute_tensor_fast(U, V, W)
        return np.allclose(T, 1.0, atol=tol)

    def search_rank(r, max_iters=50, n_restarts=3, seed=42):
        np.random.seed(seed)
        nm, mp, pn = n*m, m*p, p*n
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            lr = 0.01

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                T = compute_tensor_fast(U, V, W)
                loss = np.sum((T - 1.0)**2)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if loss < 1e-12:
                    return U, V, W

                dT = 2.0 * (T - 1.0)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for rr in range(r):
                    u = U[:, rr].reshape(n, m)
                    v = V[:, rr].reshape(m, p)
                    w = W[:, rr].reshape(p, n)
                    dU[:, rr] = np.einsum('ijk,jk,ki->ij', dT, v, w).flatten()
                    dV[:, rr] = np.einsum('ijk,ij,ki->jk', dT, u, w).flatten()
                    dW[:, rr] = np.einsum('ijk,ij,jk->ki', dT, u, v).flatten()

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                if loss > best_loss * 1.2:
                    lr *= 0.8
                else:
                    lr = min(lr * 1.02, 0.05)

        if best_loss < 1e-10:
            return best_U, best_V, best_W
        return None, None, None

    best_result = None
    for r in [24, 32, 40, 48, 56, 64]:
        if time.time() > deadline - 10:
            break
        U, V, W = search_rank(r, max_iters=40, n_restarts=2, seed=r)
        if U is not None and verify_tensor(U, V, W, tol=1e-8):
            best_result = (U, V, W)
            break

    if best_result:
        return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
