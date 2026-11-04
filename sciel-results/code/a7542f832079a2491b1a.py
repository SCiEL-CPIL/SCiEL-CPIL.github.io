import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)

    def get_strassen_decomposition():
        rank = 54
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0

        def add_strassen_block(j0, j1, k0, k1):
            nonlocal col
            # M0 = (A00+A11)(B00+B11)
            U[0*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # M1 = (A10+A11)B00
            U[1*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # M2 = A00(B01-B11)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1; V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # M3 = A11(B10-B00)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
            col += 1
            # M4 = (A00+A01)B11
            U[0*m + j0, col] += 1; U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1; W[k1*n + 0, col] += 1
            col += 1
            # M5 = (A10-A00)(B00+B01)
            U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M6 = (A01-A11)(B10+B11)
            U[0*m + j1, col] += 1; U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1

        def add_naive_block(j, k0, k1):
            nonlocal col
            U[0*m + j, col] += 1; V[j*p + k0, col] += 1; W[k0*n + 0, col] += 1; col += 1
            U[0*m + j, col] += 1; V[j*p + k1, col] += 1; W[k1*n + 0, col] += 1; col += 1
            U[1*m + j, col] += 1; V[j*p + k0, col] += 1; W[k0*n + 1, col] += 1; col += 1
            U[1*m + j, col] += 1; V[j*p + k1, col] += 1; W[k1*n + 1, col] += 1; col += 1

        for j_pair in [(0, 1), (2, 3)]:
            for k_pair in [(0, 1), (2, 3), (4, 5)]:
                add_strassen_block(j_pair[0], j_pair[1], k_pair[0], k_pair[1])
        for k_pair in [(0, 1), (2, 3), (4, 5)]:
            add_naive_block(4, k_pair[0], k_pair[1])

        return U, V, W

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1; V[j*p + k, col] = 1; W[k*n + i, col] = 1; col += 1
        return U, V, W

    U, V, W = get_strassen_decomposition()
    if check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
