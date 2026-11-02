"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

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

    r = 88
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        lr = 0.05
        for step in range(400):
            if time.time() > deadline - 5:
                break
            P = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
            resid = P - 1.0
            loss = np.sum(resid ** 2)
            if loss < 1e-12:
                break
            gU = 2 * np.einsum('ijk,jkr,kir->ijr', resid, Vr, Wr)
            gV = 2 * np.einsum('ijk,ijr,kir->jkr', resid, Ur, Wr)
            gW = 2 * np.einsum('ijk,ijr,jkr->kir', resid, Ur, Vr)
            Ur -= lr * gU
            Vr -= lr * gV
            Wr -= lr * gW
        P = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        max_err = np.max(np.abs(P - 1.0))
        if max_err < 1e-6:
            Ur_r = np.round(Ur * 2) / 2
            Vr_r = np.round(Vr * 2) / 2
            Wr_r = np.round(Wr * 2) / 2
            P_r = np.einsum('ijr,jkr,kir->ijk', Ur_r, Vr_r, Wr_r)
            if np.max(np.abs(P_r - 1.0)) < 1e-9:
                return {"U": [[float(x) for x in row] for row in Ur_r.reshape(n*m, r)],
                        "V": [[float(x) for x in row] for row in Vr_r.reshape(m*p, r)],
                        "W": [[float(x) for x in row] for row in Wr_r.reshape(p*n, r)]}
            else:
                return {"U": [[float(x) for x in row] for row in Ur.reshape(n*m, r)],
                        "V": [[float(x) for x in row] for row in Vr.reshape(m*p, r)],
                        "W": [[float(x) for x in row] for row in Wr.reshape(p*n, r)]}
    return schoolbook()
# EVOLVE-BLOCK-END
