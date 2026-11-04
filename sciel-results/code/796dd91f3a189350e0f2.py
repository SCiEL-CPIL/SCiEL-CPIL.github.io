import numpy as np
import time

DIMENSIONS = [2, 4, 4]
_START = time.time()
DEADLINE = 120.0  # safety margin below 150s kill

def _verify(U, V, W, n, m, p, trials=10, seed=0):
    try:
        U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
        if U.shape != (n*m, len(U[0])) or V.shape != (m*p, len(V[0])) or W.shape != (p*n, len(W[0])):
            return False
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        rng = np.random.default_rng(seed)
        U2 = U.reshape(n, m, rank); V2 = V.reshape(m, p, rank); W2 = W.reshape(p, n, rank)
        for _ in range(trials):
            A = rng.standard_normal((n, m)); B = rng.standard_normal((m, p))
            Ctrue = A @ B
            MA = np.einsum('ij,ijc->c', A, U2)
            MB = np.einsum('jk,jkc->c', B, V2)
            C = np.einsum('ikc,c,c->ik', W2, MA, MB)
            if not np.allclose(C, Ctrue, atol=1e-9):
                return False
        return True
    except Exception:
        return False

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block(i0, j0, k0, U, V, W, col):
    # 2x2x2 Strassen on block: i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # local A indices (li,lj), B (lj,lk), C (li,lk)
    # M1=(A00+A11)(B00+B11)
    def addU(li,lj,val): U[(i0+li)*m_global + (j0+lj)][col] += val
    ...
