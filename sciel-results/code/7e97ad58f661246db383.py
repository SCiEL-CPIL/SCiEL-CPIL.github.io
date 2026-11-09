import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _kron_decomp(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            # U: flat index (i1*n2+i2)*m + (j1*m2+j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1v = U1[i1 * m1 + j1, t1]
                    if u1v == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2v = U2[i2 * m2 + j2, t2]
                            if u2v == 0:
                                continue
                            idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            U[idx, t] = u1v * u2v
            # V: flat index (j1*m2+j2)*p + (k1*p2+k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1v = V1[j1 * p1 + k1, t1]
                    if v1v == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2v = V2[j2 * p2 + k2, t2]
                            if v2v == 0:
                                continue
                            idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V[idx, t] = v1v * v2v
            # W: flat index (k1*p2+k2)*n + (i1*n2+i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1v = W1[k1 * n1 + i1, t1]
                    if w1v == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2v = W2[k2 * n2 + i2, t2]
                            if w2v == 0:
                                continue
                            idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W[idx, t] = w1v * w2v
    return U, V, W

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _schoolbook_list(n, m, p):
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 8
    
    # Try Kronecker: (3,2,4) x (1,2,2) -> rank 48
    try:
        U1, V1, W1 = _build_schoolbook(3, 2, 4)
        U2, V2, W2 = _build_schoolbook(1, 2, 2)
        U, V, W = _kron_decomp(U1, V1, W1, 3, 2, 4, U2, V2, W2, 1, 2, 2)
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return _schoolbook_list(n, m, p)
# EVOLVE-BLOCK-END
