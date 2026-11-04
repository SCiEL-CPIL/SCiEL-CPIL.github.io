import time
import numpy as np

N, M, P = 3, 4, 8

def _verify(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Compute tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    T = np.zeros((N, M, P))
    for r in range(r):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(N):
            ui = u[i * M:(i + 1) * M]
            for k in range(P):
                T[i, :, k] += ui * v[k * M:(k + 1) * M] * w[k * N + i]
    expected = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                expected[i, j, k] = 1
    return np.allclose(T, expected, atol=1e-9)

def _build_schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}

def _build_rank88():
    rank = 88
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    c = 0
    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(M):
        for k in range(P):
            U[0 * M + j][c] = 1
            V[j * P + k][c] = 1
            W[k * N + 0][c] = 1
            c += 1
    # Part 2: i=1,2, Strassen 2x4x8
    # Split j into 2 blocks of 2, k into 4 blocks of 2
    su = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    sv = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    sw = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if su[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = js + lj
                        U[gi * M + gj][c] += su[t][l]
                    if sv[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = js + lj
                        gk = ks + lk
                        V[gj * P + gk][c] += sv[t][l]
                    if sw[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = ks + lk
                        W[gk * N + gi][c] += sw[t][l]
                c += 1
    return {'U': U, 'V': V, 'W': W}

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # Build rank-88 solution
    sol = _build_rank88()
    if _verify(sol['U'], sol['V'], sol['W']):
        return sol

    # Fallback to schoolbook
    sol = _build_schoolbook()
    return sol
