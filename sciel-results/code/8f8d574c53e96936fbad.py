import time
import numpy as np

N = 4
M = 4
P = 6

STRASSEN_A = np.array([
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1]
])

STRASSEN_B = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
])

RECON_2X2 = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 0, -1), (0, 1, 1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]


def _block_index(r, c, bs):
    return (r // bs) * bs + (r % bs), (c // bs) * bs + (c % bs)


def _fill_part1(U, V, W, col0):
    total = 0
    for outer_s in range(7):
        for inner_s in range(7):
            a_vec = STRASSEN_A[inner_s]
            b_vec = STRASSEN_B[inner_s]
            outer_a = STRASSEN_A[outer_s]
            outer_b = STRASSEN_B[outer_s]

            for a in range(4):
                ac = a_vec[a]
                if ac == 0:
                    continue
                ar = a // 2
                ac2 = a % 2
                for r in range(4):
                    rc = outer_a[r]
                    if rc == 0:
                        continue
                    gi = (r // 2) * 2 + ar
                    gj = (r % 2) * 2 + ac2
                    U[gi * M + gj, col0 + total] += ac * rc

            for b in range(4):
                bc = b_vec[b]
                if bc == 0:
                    continue
                br = b // 2
                bc2 = b % 2
                for c in range(4):
                    cc = outer_b[c]
                    if cc == 0:
                        continue
                    gj = (c // 2) * 2 + br
                    gk = (c % 2) * 2 + bc2
                    V[gj * P + gk, col0 + total] += bc * cc

            for (r1, c1, cf1) in RECON_2X2[outer_s]:
                for (r2, c2, cf2) in RECON_2X2[inner_s]:
                    gi = r1 * 2 + r2
                    gk = c1 * 2 + c2
                    W[gk * N + gi, col0 + total] += cf1 * cf2

            total += 1
    return total


def _fill_part2(U, V, W, col0):
    total = 0
    for s in range(7):
        a_vec = STRASSEN_A[s]
        b_vec = STRASSEN_B[s]
        for i in range(2):
            for j in range(2):
                for r in range(4):
                    rc = a_vec[r]
                    if rc == 0:
                        continue
                    gi = (r // 2) * 2 + i
                    gj = (r % 2) * 2 + j
                    U[gi * M + gj, col0 + total] += rc

                for c in range(4):
                    cc = b_vec[c]
                    if cc == 0:
                        continue
                    gj = (c // 2) * 2 + j
                    gk = 4 + (c % 2)
                    V[gj * P + gk, col0 + total] += cc

                for (r1, c1, cf) in RECON_2X2[s]:
                    gi = r1 * 2 + i
                    gk = 4 + c1
                    W[gk * N + gi, col0 + total] += cf

                total += 1
    return total


def _verify_tensors(U, V, W, R):
    Uf = np.asarray(U, dtype=np.float64).reshape(N, M, R)
    Vf = np.asarray(V, dtype=np.float64).reshape(M, P, R)
    Wf = np.asarray(W, dtype=np.float64).reshape(P, N, R)
    recon = np.einsum('ijc,jkc,kic->ijk', Uf, Vf, Wf)
    return np.max(np.abs(recon - 1.0))


def _schoolbook():
    r = N * M * P
    U = [[0] * r for _ in range(N * M)]
    V = [[0] * r for _ in range(M * P)]
    W = [[0] * r for _ in range(P * N)]
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][idx] = 1
                V[j * P + k][idx] = 1
                W[k * N + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _clean(mat):
    out = []
    for row in mat:
        nr = []
        for v in row:
            iv = round(v)
            if abs(v - iv) < 1e-9:
                nr.append(int(iv))
            else:
                nv = round(v, 12)
                if nv == 0:
                    nv = 0
                nr.append(nv)
        out.append(nr)
    return out


def construct():
    t_start = time.time()
    limit = t_start + 110.0

    R = 77
    U = np.zeros((N * M, R))
    V = np.zeros((M * P, R))
    W = np.zeros((P * N, R))

    n1 = _fill_part1(U, V, W, 0)
    if time.time() > limit:
        return _schoolbook()
    _fill_part2(U, V, W, n1)

    err = _verify_tensors(U, V, W, R)
    if err > 1e-6:
        return _schoolbook()

    return {"U": _clean(U), "V": _clean(V), "W": _clean(W)}
