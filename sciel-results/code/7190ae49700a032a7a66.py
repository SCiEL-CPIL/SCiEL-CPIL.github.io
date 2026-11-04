import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12

    # Build target tensor: target[a, b, ki] = 1 if (a,b,ki) corresponds to valid (i,j,k)
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                ki = k * n + i
                target[a, b, ki] = 1.0

    def check(U, V, W, tol=1e-4):
        T = np.einsum('ac,bc,ic->abi', U, V, W)
        return np.allclose(T, target, atol=tol)

    def round_half(M):
        return np.round(M * 2.0) / 2.0

    # Try ranks from small to large
    for r in range(1, 37):
        if time.time() > deadline:
            break

        for trial in range(15):
            if time.time() > deadline:
                break

            np.random.seed(123 + trial * 7 + r * 131)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)

            for it in range(300):
                if time.time() > deadline:
                    break

                # Update U: fix V, W
                VW = np.einsum('bc,ic->bic', V, W).reshape(mp * pn, r)
                for a in range(nm):
                    rhs = target[a].flatten()
                    U[a] = np.linalg.lstsq(VW, rhs, rcond=None)[0]

                # Update V: fix U, W
                UW = np.einsum('ac,ic->aic', U, W).reshape(nm * pn, r)
                for b in range(mp):
                    rhs = target[:, b].flatten()
                    V[b] = np.linalg.lstsq(UW, rhs, rcond=None)[0]

                # Update W: fix U, V
                UV = np.einsum('ac,bc->abc', U, V).reshape(nm * mp, r)
                for ki in range(pn):
                    rhs = target[:, :, ki].flatten()
                    W[ki] = np.linalg.lstsq(UV, rhs, rcond=None)[0]

                if check(U, V, W):
                    Ur, Vr, Wr = round_half(U), round_half(V), round_half(W)
                    if check(Ur, Vr, Wr):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    # Try returning unrounded if it's close enough
                    if check(U, V, W, tol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook (always valid, rank 36)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
