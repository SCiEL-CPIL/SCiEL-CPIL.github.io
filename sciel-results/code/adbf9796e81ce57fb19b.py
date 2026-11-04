import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen(i1, i2, j1, j2, k1, k2, offset):
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, offset] = 1; U[i2*m+j2, offset] = 1
        V[j1*p+k1, offset] = 1; V[j2*p+k2, offset] = 1
        W[k1*n+i1, offset] = 1; W[k2*n+i2, offset] = 1
        offset += 1

        # M2 = (A10 + A11) B00
        U[i2*m+j1, offset] = 1; U[i2*m+j2, offset] = 1
        V[j1*p+k1, offset] = 1
        W[k1*n+i2, offset] = 1; W[k2*n+i2, offset] = -1
        offset += 1

        # M3 = A00 (B01 - B11)
        U[i1*m+j1, offset] = 1
        V[j1*p+k2, offset] = 1; V[j2*p+k2, offset] = -1
        W[k2*n+i1, offset] = 1; W[k2*n+i2, offset] = 1
        offset += 1

        # M4 = A11 (B10 - B00)
        U[i2*m+j2, offset] = 1
        V[j2*p+k1, offset] = 1; V[j1*p+k1, offset] = -1
        W[k1*n+i1, offset] = 1; W[k1*n+i2, offset] = 1
        offset += 1

        # M5 = (A00 + A01) B11
        U[i1*m+j1, offset] = 1; U[i1*m+j2, offset] = 1
        V[j2*p+k2, offset] = 1
        W[k1*n+i1, offset] = -1; W[k2*n+i1, offset] = 1
        offset += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, offset] = 1; U[i1*m+j1, offset] = -1
        V[j1*p+k1, offset] = 1; V[j1*p+k2, offset] = 1
        W[k2*n+i2, offset] = 1
        offset += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, offset] = 1; U[i2*m+j2, offset] = -1
        V[j2*p+k1, offset] = 1; V[j2*p+k2, offset] = 1
        W[k1*n+i1, offset] = 1
        offset += 1

        return offset

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    offset = 0
    offset = add_strassen(0, 1, 0, 1, 0, 1, offset)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    offset = add_strassen(0, 1, 0, 1, 2, 3, offset)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4}
    # Rank 1: U=1, V=1, W=1
    for i in [0, 1]:
        for j in [0, 1]:
            U[i*m+j, offset] = 1
            V[j*p+4, offset] = 1
            W[4*n+i, offset] = 1
    offset += 1
    
    # Block 4: i in {0,1}, j in {2}, k in {0..4}
    # Rank 1: U=1, V=1, W=1
    for i in [0, 1]:
        for k in range(p):
            U[i*m+2, offset] = 1
            V[2*p+k, offset] = 1
            W[k*n+i, offset] = 1
    offset += 1
    
    # Verify
    def verify(U, V, W, rank):
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(tensor, 1.0, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails (should not happen)
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

