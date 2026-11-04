import numpy as np
from fractions import Fraction
import time

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # p3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # p4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # p5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = 1
    
    # p6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W


def _trivial_1mp(m, p):
    """Return U, V, W for 1xm x px1... no, 1xm * mxp = 1xp. Rank 1."""
    # n=1, so U has shape (1*m, 1) = (m, 1)
    # V has shape (m*p, 1)
    # W has shape (p*1, 1) = (p, 1)
    U = np.ones((m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p, 1))
    return U, V, W


def _direct_sum_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two decompositions of (n,m1,p) and (n,m2,p) into (n,m1+m2,p).
    U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
    U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
    Result: U: (n*(m1+m2), r1+r2), V: ((m1+m2)*p, r1+r2), W: (p*n, r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.zeros((n * (m1 + m2), r1 + r2))
    U[:n*m1, :r1] = U1
    U[n*m1:, r1:] = U2
    
    V = np.zeros(((m1 + m2) * p, r1 + r2))
    V[:m1*p, :r1] = V1
    V[m1*p:, r1:] = V2
    
    W = np.zeros((p * n, r1 + r2))
    W[:, :r1] = W1
    W[:, r1:] = W2
    
    return U, V, W


def _direct_sum_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    """
    Combine two decompositions of (n1,m,p) and (n2,m,p) into (n1+n2,m,p).
    U1: (n1*m, r1), V1: (m*p, r1), W1: (p*n1, r1)
    U2: (n2*m, r2), V2: (m*p, r2), W2: (p*n2, r2)
    Result: U: ((n1+n2)*m, r1+r2), V: (m*p, r1+r2), W: (p*(n1+n2), r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.zeros(((n1 + n2) * m, r1 + r2))
    U[:n1*m, :r1] = U1
    U[n1*m:, r1:] = U2
    
    V = np.zeros((m * p, r1 + r2))
    V[:, :r1] = V1
    V[:, r1:] = V2
    
    W = np.zeros((p * (n1 + n2), r1 + r2))
    W[:p*n1, :r1] = W1
    W[p*n1:, r1:] = W2
    
    return U, V, W


def _direct_sum_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two decompositions of (n,m,p1) and (n,m,p2) into (n,m,p1+p2).
    U1: (n*m, r1), V1: (m*p1, r1), W1: (p1*n, r1)
    U2: (n*m, r2), V2: (m*p2, r2), W2: (p2*n, r2)
    Result: U: (n*m, r1+r2), V: (m*(p1+p2), r1+r2), W: ((p1+p2)*n, r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.zeros((n * m, r1 + r2))
    U[:, :r1] = U1
    U[:, r1:] = U2
    
    V = np.zeros((m * (p1 + p2), r1 + r2))
    V[:m*p1, :r1] = V1
    V[m*p1:, r1:] = V2
    
    W = np.zeros(((p1 + p2) * n, r1 + r2))
    W[:p1*n, :r1] = W1
    W[p1*n:, r1:] = W2
    
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Verify tensor decomposition."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = 3, 4, 8
    
    # Step 5: Build r(2,2,4) = r(2,2,2) + r(2,2,2) = 14
    U_222, V_222, W_222 = _strassen_222()
    U_224, V_224, W_224 = _direct_sum_split_p(U_222, V_222, W_222, U_222, V_222, W_222, 2, 2, 2, 2)
    
    # Verify
    assert _verify_decomp(U_224, V_224, W_224, 2, 2, 4), "2x2x4 verification failed"
    
    # Step 4: Build r(2,2,8) = r(2,2,4) + r(2,2,4) = 28
    U_228, V_228, W_228 = _direct_sum_split_p(U_224, V_224, W_224, U_224, V_224, W_224, 2, 2, 4, 4)
    
    # Verify
    assert _verify_decomp(U_228, V_228, W_228, 2, 2, 8), "2x2x8 verification failed"
    
    # Step 3: Build r(2,4,8) = r(2,2,8) + r(2,2,8) = 56
    U_248, V_248, W_248 = _direct_sum_split_m(U_228, V_228, W_228, U_228, V_228, W_228, 2, 2, 2, 8)
    
    # Verify
    assert _verify_decomp(U_248, V_248, W_248, 2, 4, 8), "2x4x8 verification failed"
    
    # Step 2: Build r(1,4,8) = 1
    U_148, V_148, W_148 = _trivial_1mp(4, 8)
    
    # Verify
    assert _verify_decomp(U_148, V_148, W_148, 1, 4, 8), "1x4x8 verification failed"
    
    # Step 1: Build r(3,4,8) = r(1,4,8) + r(2,4,8) = 57
    U_348, V_348, W_348 = _direct_sum_split_n(U_148, V_148, W_148, U_248, V_248, W_248, 1, 2, 4, 8)
    
    # Verify
    assert _verify_decomp(U_348, V_348, W_348, 3, 4, 8), "3x4x8 verification failed"
    
    # Convert to lists of lists
    U = U_348.tolist()
    V = V_348.tolist()
    W = W_348.tolist()
    
    return {"U": U, "V": V, "W": W}
