import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-8):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def solve_for_W(U, V):
        rank = U.shape[1]
        W = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in 0..m-1
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = x
        return W

    def solve_for_U(V, W):
        rank = V.shape[1]
        U = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = x
        return U

    def solve_for_V(U, W):
        rank = U.shape[1]
        V = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = x
        return V

    fallback = schoolbook()

    for rank in range(15, 35):
        if time.time() > deadline - 20:
            break
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            rng = np.random.default_rng(987654 + rank * 10000 + trial)
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))

            for it in range(200):
                if time.time() > deadline - 10:
                    break
                U_new = solve_for_U(V, W)
                V_new = solve_for_V(U_new, W)
                W_new = solve_for_W(U_new, V_new)

                diff = (np.max(np.abs(W_new - W)) + np.max(np.abs(V_new - V)) +
                        np.max(np.abs(U_new - U)))
                U, V, W = U_new, V_new, W_new
                if diff < 1e-12:
                    break

            if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return fallback
