"""Find rank-minimal exact tensor decomposition for 2x2x9 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product
import time

DIMENSIONS = [2, 2, 9]

def _build_target_tensor(n, m, p):
    """The matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k."""
    return np.ones((n, m, p), dtype=float)

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W decompose the target tensor."""
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            ui = U[i*m + j if False else i*m]  # need to be careful
        # Let's do it properly
    # Rebuild
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    return np.allclose(T, np.ones((n,m,p)), atol=tol)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fractions."""
    rank = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook gives rank 36
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
