import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100  # 20s margin

    # Schoolbook rank 18
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for target_rank in range(rank_sb - 1, 12, -1):
        if time.time() > deadline - 10:
            break

        found = False
        for restart in range(500):
            if time.time() > deadline - 10:
                break

            seed = 987654321 + restart * 6151 + target_rank * 104729
            np.random.seed(seed)

            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)

            # Normalize each column
            for r in range(target_rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale

            lr = 1.0
            for step in range(10000):
                if time.time() > deadline - 10:
                    break

                err = 0.0
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)

                for (i, j, k) in indices:
                    ui = U[i*m+j, :]
                    vj = V[j*p+k, :]
                    wk = W[k*n+i, :]
                    s = np.dot(wk, ui * vj)
                    e = s - 1.0
                    err += e * e
                    f = 2.0 * e
                    gU[i*m+j, :] += f * wk * vj
                    gV[j*p+k, :] += f * wk * ui
                    gW[k*n+i, :] += f * ui * vj

                if err < 1e-25:
                    break

                norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
                if norm_g < 1e-15:
                    break

                step_size = min(lr, 0.3 / norm_g)
                U -= step_size * gU
                V -= step_size * gV
                W -= step_size * gW

                if step % 1000 == 999:
                    lr *= 0.7

            if verify(U, V, W):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if found:
            break

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
