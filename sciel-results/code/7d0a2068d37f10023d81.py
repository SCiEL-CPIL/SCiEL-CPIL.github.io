import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 8
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # For each column j in A (0 to m-1)
    for j in range(m):
        # Term for row 0 of A
        r0 = j
        # U index: i=0, col=j -> 0*m + j
        U[0 * m + j, r0] = 1
        # V index: row=j, col=k -> j*p + k
        for k in range(p):
            V[j * p + k, r0] = 1
        # W index: col=k, row=i=0 -> k*n + 0
        for k in range(p):
            W[k * n + 0, r0] = 1
            
        # Term for row 1 of A
        r1 = m + j
        # U index: i=1, col=j -> 1*m + j
        U[1 * m + j, r1] = 1
        # V index: row=j, col=k -> j*p + k
        for k in range(p):
            V[j * p + k, r1] = 1
        # W index: col=k, row=i=1 -> k*n + 1
        for k in range(p):
            W[k * n + 1, r1] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

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
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
