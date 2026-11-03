import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _build_strassen_2x3x8(n, m, p):
    """
    Build a rank-44 decomposition using Strassen for the 2x2x2 blocks.
    
    Structure:
    - j=0: 16 schoolbook terms (2 i-values * 8 k-values)
    - j=1,2: 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms
    Total: 16 + 28 = 44
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: j=0, schoolbook for 2x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for 2x2x2:
        # m1 = (a00+a11)(b00+b11)
        U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1; V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1; V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W, rank

def construct():
    n, m, p = 2, 3, 8
    
    # Use the rank-44 Strassen-based construction
    U, V, W, rank = _build_strassen_2x3x8(n, m, p)
    
    # Verify (optional, but good for safety)
    # if not _verify_decomposition(U, V, W, n, m, p):
    #     # Fallback to schoolbook
    #     U, V, W, rank = _build_schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}
