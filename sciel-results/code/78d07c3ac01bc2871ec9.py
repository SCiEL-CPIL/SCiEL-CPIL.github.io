"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _winograd_2x2x3():
    """Winograd rank-7 decomposition for 2x2 * 2x3.
    Returns U (2x2, 7), V (2x3, 7), W (3x2, 7) where:
      U[i,j,c] = (i-a)*(j-b), V[j,k,c] = (a-j)*(b-k), W[i,k,c] = 1
    with (a,b) pairs: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,3)
    """
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 3)]
    rank = len(pairs)
    U = [[0] * rank for _ in range(2 * 2)]   # rows: i*2+j
    V = [[0] * rank for _ in range(2 * 3)]   # rows: j*3+k
    W = [[0] * rank for _ in range(3 * 2)]   # rows: k*2+i
    for c, (a, b) in enumerate(pairs):
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = (i - a) * (j - b)
        for j in range(2):
            for k in range(3):
                V[j * 3 + k][c] = (a - j) * (b - k)
        for i in range(2):
            for k in range(3):
                W[k * 2 + i][c] = 1
    return U, V, W


def _kronecker_4x4x9():
    """Kronecker product of two Winograd 2x2x3 decompositions.
    4x4 * 4x9 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x3)
    Total rank = 7 * 7 = 49.
    """
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 7 * 7
    n, m, p = 4, 4, 9
    U = [[0] * rank for _ in range(n * m)]   # 16 x 49
    V = [[0] * rank for _ in range(m * p)]   # 36 x 49
    W = [[0] * rank for _ in range(p * n)]   # 36 x 49
    
    # Indexing: A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    # U row: i*m+j = (i1*2+i2)*4 + (j1*2+j2)
    # V row: j*p+k = (j1*2+j2)*9 + (k1*3+k2)
    # W row: k*n+i = (k1*3+k2)*4 + (i1*2+i2)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1 * 2 + j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2 * 2 + j2][c2]
                            if u2 == 0:
                                continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * m + j][c] = u1 * u2
            
            for j1 in range(2):
                for k1 in range(3):
                    v1 = V1[j1 * 3 + k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2 * 3 + k2][c2]
                            if v2 == 0:
                                continue
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            V[j * p + k][c] = v1 * v2
            
            for i1 in range(2):
                for k1 in range(3):
                    w1 = W1[k1 * 2 + i1][c1]
                    if w1 == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(3):
                            w2 = W2[k2 * 2 + i2][c2]
                            if w2 == 0:
                                continue
                            i = i1 * 2 + i2
                            k = k1 * 3 + k2
                            W[k * n + i][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}


def _rank36(n, m, p):
    """Attempt rank-36 decomposition: rank = max(n*m, m*p, p*n) = 36.
    Column c indexes (j0 = c mod m, k0 = c mod p).
    """
    rank = max(n * m, m * p, p * n)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        j0 = c % m
        k0 = c % p
        i0 = c % n
        for i in range(n):
            if j == j0 if False else True:
                val = -1 if i == 0 else 1
                U[i * m + j0][c] = val
        for j in range(m):
            val = -1 if j == 0 else 1
            V[j * p + k0][c] = val
        W[k0 * n + i0][c] = 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try candidates in order of increasing rank; return first valid one.
    candidates = []
    
    # Rank 49: Kronecker Winograd
    if time.time() < _DEADLINE - 10:
        try:
            cand49 = _kronecker_4x4x9()
            candidates.append((49, cand49))
        except Exception:
            pass
    
    # Rank 144: Schoolbook (always valid)
    cand_sb = _schoolbook(n, m, p)
    candidates.append((n * m * p, cand_sb))
    
    # Sort by rank (lower is better)
    candidates.sort(key=lambda x: x[0])
    
    for rank_val, cand in candidates:
        if time.time() > _DEADLINE - 5:
            break
        try:
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                return cand
        except Exception:
            continue
    
    # Fallback: return schoolbook
    return cand_sb

# EVOLVE-BLOCK-END
