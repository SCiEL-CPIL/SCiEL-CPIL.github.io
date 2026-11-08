import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = n * m * p  # 2 * 4 * 4 = 32

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # r maps to (i, j, k) where r = i*m*p + j*p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k

                # U: u = i*m + j. Non-zero only for matching i, j
                u_idx = i * m + j
                U[u_idx, r] = 1.0

                # V: v = j*p + k. Non-zero only for matching j, k
                v_idx = j * p + k
                V[v_idx, r] = 1.0

                # W: w = k*n + i. Non-zero only for matching k, i
                w_idx = k * n + i
                W[w_idx, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
