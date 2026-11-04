import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    return terms

def _build_decomposition(n, m, p):
    rank = 39
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    st = _strassen_2x2x2()
    col = 0
    
    for ks in (0, 2, 4):
        for (uv, vv, wv) in st:
            ag = [ks, ks+1, m+ks, m+ks+1]
            bg = [ks, ks+1, p+ks, p+ks+1]
            cg = [ks*n, ks*n+1, (ks+1)*n, (ks+1)*n+1]
            for t in range(4):
                if uv[t]: U[ag[t]][col] += uv[t]
                if vv[t]: V[bg[t]][col] += vv[t]
                if wv[t]: W[cg[t]][col] += wv[t]
            col += 1
    
    # k=6: 2x2x1, rank 2
    # C[6,i] = sum_j A[i,j]B[j,6], j in {0,1}
    # term: A[0,6]*B[0,6] -> C[6,0]+C[6,1]
    U[6][col] += 1; V[6][col] += 1; W[12][col] += 1; W[13][col] += 1
    col += 1
    # term: A[1,6]*B[1,6] -> C[6,0]+C[6,1]
    U[m+6][col] += 1; V[p+6][col] += 1; W[12][col] += 1; W[13][col] += 1
    col += 1
    
    # j=2: 2x1x7, rank 14
    for k in range(7):
        U[2][col] += 1; V[2*p+k][col] += 1; W[k*n][col] += 1
        col += 1
        U[m+2][col] += 1; V[2*p+k][col] += 1; W[k*n+1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p):
    r = len(U[0])
    ua = np.array(U, dtype=np.float64)
    va = np.array(V, dtype=np.float64)
    wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(ua[i*m+j, rr] * va[j*p+k, rr] * wa[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    sol = _build_decomposition(n, m, p)
    if _check(sol["U"], sol["V"], sol["W"], n, m, p):
        return sol
    # fallback rank 42
    r = n*m*p
    U = [[Fraction(0)]*r for _ in range(n*m)]
    V = [[Fraction(0)]*r for _ in range(m*p)]
    W = [[Fraction(0)]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
