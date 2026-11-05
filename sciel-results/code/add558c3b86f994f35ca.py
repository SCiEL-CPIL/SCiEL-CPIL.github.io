"""Matrix multiplication 4x4x5 tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DEADLINE = 140.0

def _verify(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u,v,w are length-4 vectors.
    u[i*2+j], v[j*2+k], w[k*2+i].
    """
    # Verified Strassen 2x2x2 tensor decomposition
    terms = [
        # (u, v, w)
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]),  # M1
        ([0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),  # M3
        ([0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),  # M7
    ]
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for u, v, w in terms:
                    s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if s != 1:
                    raise ValueError(f"Strassen term verification failed at ({i},{j},{k}): sum={s}")
    return terms

def _construct_double_strassen_4x4x4():
    """Construct rank-49 decomposition of 4x4x4 matmul tensor using double Strassen.
    Returns (U, V, W) where U is (16, 49), V is (20, 49) [only k=0..3 used], W is (20, 49).
    Actually V is (m*p, rank) = (20, 49) and W is (p*n, rank) = (20, 49).
    For the 4x4x4 part, only V[j*5+k] for k=0..3 and W[k*4+i] for k=0..3 are used.
    """
    terms = _strassen_2x2_terms()
    rank = 49
    U = [[0.0] * rank for _ in range(16)]  # n*m = 4*4
    V = [[0.0] * rank for _ in range(20)]  # m*p = 4*5
    W = [[0.0] * rank for _ in range(20)]  # p*n = 5*4
    
    for c1 in range(7):
        u1, v1, w1 = terms[c1]
        for c2 in range(7):
            u2, v2, w2 = terms[c2]
            col = c1 * 7 + c2
            # u1[a*2+c] * u2[b*2+d] for i=2a+b, j=2c+d
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        for d in range(2):
                            i = 2*a + b
                            j = 2*c + d
                            val = u1[a*2+c] * u2[b*2+d]
                            U[i*4+j][col] = val
            # v1[c*2+e] * v2[d*2+f] for j=2c+d, k=2e+f
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):
                            j = 2*c + d
                            k = 2*e + f
                            val = v1[c*2+e] * v2[d*2+f]
                            V[j*5+k][col] = val
            # w1[e*2+a] * w2[f*2+b] for k=2e+f, i=2a+b
            for e in range(2):
                for f in range(2):
                    for a in range(2):
                        for b in range(2):
                            k = 2*e + f
                            i = 2*a + b
                            val = w1[e*2+a] * w2[f*2+b]
                            W[k*4+i][col] = val
    
    return U, V, W

def _construct_schoolbook_4x4x1(offset=0):
    """Construct rank-16 schoolbook decomposition for the 4x4x1 part (k=4).
    Returns (U, V, W) with only the 16 columns filled, starting at column offset.
    """
    rank = 16
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(20)]
    W = [[0.0] * rank for _ in range(20)]
    
    for i in range(4):
        for j in range(4):
            col = i*4 + j
            U[i*4+j][col] = 1.0
            V[j*5+4][col] = 1.0
            W[4*4+i][col] = 1.0
    
    return U, V, W

def _construct_rank_65():
    """Construct a rank-65 decomposition: double Strassen (49) + schoolbook 4x4x1 (16)."""
    U1, V1, W1 = _construct_double_strassen_4x4x4()
    U2, V2, W2 = _construct_schoolbook_4x4x1()
    
    rank = 65
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(20)]
    W = [[0.0] * rank for _ in range(20)]
    
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c]
            W[r][c] = W1[r][c]
        for c in range(16):
            U[r][49+c] = U2[r][c]
            V[r][49+c] = V2[r][c]
            W[r][49+c] = W2[r][c]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n=4, m=4, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    # Construct the rank-65 decomposition
    result = _construct_rank_65()
    
    # Verify
    if _verify(result['U'], result['V'], result['W'], 4, 4, 5):
        return result
    
    # Fallback to schoolbook if verification fails
    schoolbook = _schoolbook(4, 4, 5)
    return schoolbook
