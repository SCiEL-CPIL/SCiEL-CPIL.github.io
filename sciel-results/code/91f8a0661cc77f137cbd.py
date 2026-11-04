"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

_T0 = time.time()
_DEADLINE = 120.0

def _mul_tensor(n, m, p):
    """T[i,j,k] = 1 iff (i,j,k) is a schoolbook term."""
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Check U * V * W == multiplication tensor. Returns bool."""
    T = _mul_tensor(n, m, p)
    r = len(U[0])
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # Reconstruct tensor R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    R = np.zeros((n, m, p), dtype=object)
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # expand u into (n,m), v into (m,p), w into (p,n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] = R[i, j, k] + uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if R[i, j, k] != T[i, j, k]:
                    return False
    return True

def _strassen2x2x2():
    """Return (U,V,W) for 2x2x2 Strassen, rank 7, indices (i,j,k) -> flat."""
    # m=2, n=2, p=2. U index i*2+j, V index j*2+k, W index k*2+i.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    def set(c, u, v, w):
        for (a, b) in u: U[a*2+b][c] += 1
        for (a, b) in v: V[a*2+b][c] += 1
        for (a, b) in w: W[a*2+b][c] += 1
    # c0 = (A00)(B00+B11)
    set(0, [(0,0)], [(0,0),(1,1)], [(0,0),(1,1)])
    # c1 = (A10+A11)(B00)
    set(1, [(1,0),(1,1)], [(0,0)], [(0,0)])
    # c2 = (A00)(B10+B11)
    set(2, [(0,0)], [(1,0),(1,1)], [(0,1),(1,1)])
    # c3 = (A10)(B01-B10)
    set(3, [(1,0)], [(0,1),(-1,0)], [(0,1),(1,0)])
    # c4 = (A00+A11)(B10)
    set(4, [(0,0),(1,1)], [(1,0)], [(1,0)])
    # c5 = (A01-A10)(B00+B10)
    set(5, [(0,1),(-1,0)], [(0,0),(1,0)], [(0,0),(1,0)])
    # c6 = (A11-A01)(B01+B11)
    set(6, [(1,1),(-0,1)], [(0,1),(1,1)], [(0,1),(1,1)])
    return U, V, W

def _block_decomp():
    """Build rank-33 decomposition for 2x3x6.
    Split m=3 into [2,1].
      Block A (m=2): 2x2x6 = 2x2x2 (Strassen) kron 1x1x3 (rank 3) -> rank 21
      Block B (m=1): 2x1x6 -> rank 12 schoolbook
    Total 33.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0]*rank for _ in range(n*m)]   # 6 rows
    V = [[0]*rank for _ in range(m*p)]   # 18 rows
    W = [[0]*rank for _ in range(p*n)]   # 12 rows

    # ---- Block A: j in {0,1}, k in {0..5}, i in {0,1} ----
    sU, sV, sW = _strassen2x2x2()
    # sU index i*2+j (j in 0,1); sV index j*2+k (k in 0,1); sW index k*2+i (i in 0,1)
    # Kronecker with 1x1x3: split k (0..5) into (kk in 0,1) and (t in 0,1,2), k = kk*3 + t
    for c in range(7):
        for t in range(3):
            col = c*3 + t
            # U: i*2 + j  -> global i*m + j = i*3 + j
            for i in range(2):
                for j in range(2):
                    U[i*3 + j][col] += sU[i*2 + j][c]
            # V: j*2 + kk  -> global j*p + k = j*6 + (kk*3 + t)
            for j in range(2):
                for kk in range(2):
                    V[j*6 + kk*3 + t][col] += sV[j*2 + kk][c]
            # W: kk*2 + i  -> global k*n + i = (kk*3 + t)*2 + i
            for kk in range(2):
                for i in range(2):
                    W[(kk*3 + t)*2 + i][col] += sW[kk*2 + i][c]

    # ---- Block B: j=2, k in 0..5, i in 0..1 (schoolbook, rank 12) ----
    col = 21
    for i in range(2):
        for k in range(6):
            U[i*3 + 2][col] = 1
            V[2*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    return U, V, W

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    try:
        U, V, W = _block_decomp()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
