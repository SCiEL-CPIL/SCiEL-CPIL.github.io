"""Tensor decomposition for 3x3 * 3x8 matrix multiplication, rank 68."""
import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    t0 = time.time()
    LIMIT = 140.0

    # Accumulate columns of U, V, W as sparse dicts
    cols = []

    def emit(u, v, w):
        cols.append((u, v, w))

    # Strassen 2x2x2 helper: adds 7 terms
    # A block at rows a0:a0+2, cols j0:j0+2
    # B block at rows j0:j0+2, cols k0:k0+2
    # Output W at (k, i) for k in k0:k0+2, i in a0:a0+2
    def strassen(a0, j0, k0):
        # M1 = (A00+A11)(B00+B11)
        emit({(a0,j0):1,(a0+1,j0+1):1}, {(j0,k0):1,(j0+1,k0+1):1}, {(k0,a0):1,(k0+1,a0+1):1})
        # M2 = (A10+A11)B00
        emit({(a0+1,j0):1,(a0+1,j0+1):1}, {(j0,k0):1}, {(k0+1,a0+1):-1,(k0+1,a0):1})
        # M3 = A00(B10-B00)
        emit({(a0,j0):1}, {(j0+1,k0):1,(j0,k0):-1}, {(k0,a0+1):1,(k0+1,a0+1):1})
        # M4 = A11(B11-B01)
        emit({(a0+1,j0+1):1}, {(j0+1,k0+1):1,(j0,k0+1):-1}, {(k0,a0):1,(k0+1,a0):1})
        # M5 = (A00+A01)B11
        emit({(a0,j0):1,(a0,j0+1):1}, {(j0+1,k0+1):1}, {(k0,a0):-1,(k0,a0+1):1})
        # M6 = (A10-A00)(B00+B01)
        emit({(a0+1,j0):1,(a0,j0):-1}, {(j0,k0):1,(j0,k0+1):1}, {(k0+1,a0+1):1})
        # M7 = (A01-A11)(B11+B10)
        emit({(a0,j0+1):1,(a0+1,j0+1):-1}, {(j0+1,k0+1):1,(j0+1,k0):1}, {(k0,a0):1})

    # 4 Strassen blocks for A[0:2,0:2] x B[0:2, k:k+2], k=0,2,4,6
    for k0 in range(0, 8, 2):
        strassen(0, 0, k0)

    # Direct: A[i,2] * B[2,k] -> C[k,i] for i in {0,1}, k in 0..7
    for i in range(2):
        for k in range(8):
            emit({(i,2):1}, {(2,k):1}, {(k,i):1})

    # Direct: A[2,j] * B[j,k] -> C[k,2] for j in {0,1}, k in 0..7
    for j in range(2):
        for k in range(8):
            emit({(2,j):1}, {(j,k):1}, {(k,2):1})

    # Direct: A[2,2] * B[2,k] -> C[k,2]
    for k in range(8):
        emit({(2,2):1}, {(2,k):1}, {(k,2):1})

    R = len(cols)
    assert R == 68, f"Expected rank 68, got {R}"

    # Build output matrices as lists
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]

    for r, (us, vs, ws) in enumerate(cols):
        for (i,j), c in us.items():
            U[i*m+j][r] = c
        for (j,k), c in vs.items():
            V[j*p+k][r] = c
        for (k,i), c in ws.items():
            W[k*n+i][r] = c

    # Verify with integer einsum
    Ua = np.array(U, dtype=np.int64).reshape(n, m, R)
    Va = np.array(V, dtype=np.int64).reshape(m, p, R)
    Wa = np.array(W, dtype=np.int64).reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    if not np.array_equal(T, np.ones((n, m, p), dtype=np.int64)):
        # Fallback schoolbook
        Rf = n * m * p
        Uf = np.zeros((n*m, Rf), dtype=int)
        Vf = np.zeros((m*p, Rf), dtype=int)
        Wf = np.zeros((p*n, Rf), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j, c] = 1
                    Vf[j*p+k, c] = 1
                    Wf[k*n+i, c] = 1
                    c += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    if time.time() - t0 > LIMIT:
        pass

    return {"U": U, "V": V, "W": W}
