import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150 s

    n, m, p = 4, 4, 5

    # ---- Winograd 2x2x2 factors (rank 7, integer) ----
    # U row (i,j), V row (j',k), W row (k',i'); tensor T[(i,j),(j',k),(k',i')]=delta_{j,j'} delta_{i,i'}
    U2 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
    ], dtype=int)
    V2 = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ], dtype=int)
    W2 = np.array([
        [1, -1, 1, -1, 1],
        [1, -1, 1, 0, -1],
        [1, 1, -1, 1, 0],
        [1, 1, -1, 0, 1],
    ], dtype=int)

    def verify(U, V, W):
        # sum_c U[i,j,c] V[j',k,c] W[k',i',c]  ==  delta_{j,j'} delta_{i,i'}
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return bool(np.allclose(res, 1.0, atol=1e-9))

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ---- Build 4x4x4 core via Kronecker (rank 49) ----
    U4k = np.kron(U2, U2)  # 16 x 49
    V4k = np.kron(V2, V2)
    W4k = np.kron(W2, W2)
    perm = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1, col1 = r1 // 2, r1 % 2
        for r2 in range(4):
            row2, col2 = r2 // 2, r2 % 2
            kron_idx = r1 * 4 + r2
            i = row1 * 2 + row2
            j = col1 * 2 + col2
            perm[i * 4 + j] = kron_idx
    U4 = U4k[perm, :]
    V4 = V4k[perm, :]
    W4 = W4k[perm, :]

    # ---- 5th-column handling ----
    # Schoolbook: 16 terms.
    U5 = np.zeros((16, 16), dtype=int)
    V5 = np.zeros((20, 16), dtype=int)
    W5 = np.zeros((20, 16), dtype=int)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U5[i * 4 + j, col] = 1
            V5[j * 5 + 4, col] = 1
            W5[4 * 4 + i, col] = 1

    def assemble(r5, U5x, V5x, W5x):
        rank = 49 + r5
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((20, rank), dtype=int)
        W = np.zeros((20, rank), dtype=int)
        U[:, 0:49] = U4
        V[:16, 0:49] = V4
        W[:16, 0:49] = W4
        U[:, 49:] = U5x
        V[:, 49:] = V5x
        W[:, 49:] = W5x
        return U, V, W

    def to_list(M):
        return [[int(M[r, c]) for c in range(M.shape[1])] for r in range(M.shape[0])]

    # Candidate with 5th column rank 16
    U, V, W = assemble(16, U5, V5, W5)
    if verify(U, V, W):
        best_rank, best_U, best_V, best_W = 65, U, V, W
    else:
        U, V, W = schoolbook()
        best_rank, best_U, best_V, best_W = 80, U, V, W

    # ---- Try to reduce 5th-column rank ----
    # (a) exact integer search: M (4x4), u (4,), w (4,), w[i]=1; need M @ u == ones(4)
    found_a = False
    t0 = time.time()
    for vals in itertools.product([-2, -1, 0, 1, 2], repeat=16):
        M = np.array(vals, dtype=int).reshape(4, 4)
        sol = np.linalg.lstsq(M, np.ones(4, dtype=int), rcond=None)[0]
        if np.all(np.abs(sol - np.round(sol)) < 1e-6) and np.max(np.abs(np.round(sol))) <= 2:
            u = np.round(sol).astype(int)
            if np.array_equal(M @ u, np.ones(4, dtype=int)):
                w = np.ones(4, dtype=int)
                r5 = 4
                U5x = np.zeros((16, r5), dtype=int)
                V5x = np.zeros((20, r5), dtype=int)
                W5x = np.zeros((20, r5), dtype=int)
                for t in range(4):
                    U5x[t * 4:(t + 1) * 4, t] = 1
                    for j in range(4):
                        V5x[j * 5 + 4, t] = M[j, t]
                    W5x[4 * 4 + t, t] = w[t]
                U2, V2x, W2x = assemble(r5, U5x, V5x, W5x)
                if verify(U2, V2x, W2x):
                    best_rank, best_U, best_V, best_W = 49 + r5, U2, V2x, W2x
                    found_a = True
                    break
        if time.time() > deadline - 25:
            break
    if found_a or time.time() > deadline - 25:
        pass
    else:
        # (b) 4x4x1 tensor search over integer factors, rank 4..7
        # T[i, j, 0] = 1 = sum_t U[i,t] V[j,t] W[0,t]
        t0 = time.time()
        for r5 in range(4, 8):
            if time.time() > deadline - 20:
                break
            # enumerate W[0,:] in {-1,0,1}^r5, require gcd 1
            for wvals in itertools.product([-1, 0, 1], repeat=r5):
                wv = np.array(wvals, dtype=int)
                if np.gcd.reduce(np.abs(wv)) != 1:
                    continue
                # enumerate U (4 x r5) and V (4 x r5) in {-1,0,1}
                for uvals in itertools.product([-1, 0, 1], repeat=4 * r5):
                    if time.time() > deadline - 20:
                        break
                    Uu = np.array(uvals, dtype=int).reshape(4, r5)
                    for vvals in itertools.product([-1, 0, 1], repeat=4 * r5):
                        Vv = np.array(vvals, dtype=int).reshape(4, r5)
                        # need (Uu * wv) @ Vv.T == ones(4,4)
                        if np.array_equal((Uu * wv) @ Vv.T, np.ones(4, dtype=int)):
                            U5x = np.zeros((16, r5), dtype=int)
                            V5x = np.zeros((20, r5), dtype=int)
                            W5x = np.zeros((20, r5), dtype=int)
                            for i in range(4):
                                U5x[i * 4:(i + 1) * 4, :] = Uu[i, :]
                            for j in range(4):
                                V5x[j * 5 + 4, :] = Vv[j, :]
                            W5x[4 * 4, :] = wv
                            U2, V2x, W2x = assemble(r5, U5x, V5x, W5x)
                            if verify(U2, V2x, W2x):
                                best_rank, best_U, best_V, best_W = 49 + r5, U2, V2x, W2x
                                break
                    if best_rank <= 49 + r5:
                        break
                if best_rank <= 49 + r5:
                    break
            if best_rank <= 49 + r5:
                break

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
