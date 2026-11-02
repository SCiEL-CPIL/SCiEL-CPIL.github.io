"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank26_decomposition():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    
    Partition: A = [[A11, a1], [a2, a3]] where A11 is 2x2, a1 is 2x1, a2 is 1x2, a3 is 1x1
               B = [[B11, b1], [b2, b3]] where B11 is 2x2, b1 is 2x1, b2 is 1x2, b3 is 1x1
               C = [[C11, c1], [c2, c3]] where C11 is 2x2, c1 is 2x1, c2 is 1x2, c3 is 1x1
    
    C11 = A11*B11 + a1*b2  (2x2)
    c1  = A11*b1 + a1*b3   (2x1)
    c2  = a2*B11 + a3*b2   (1x2)
    c3  = a2*b1 + a3*b3    (1x1)
    
    Use Strassen (rank 7) for A11*B11, schoolbook for all other terms.
    Cross terms: a1*b2 (4), A11*b1 (4), a1*b3 (2), a2*B11 (4), a3*b2 (2), a2*b1 (2), a3*b3 (1) = 19
    Total: 7 + 19 = 26
    """
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    rank = 26
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    # Strassen for A11*B11 (A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; contributes to C rows 0,1 cols 0,1)
    # M1 = (A11[0,0]+A11[1,1])*(B11[0,0]+B11[1,1]) -> contributes to C11[0,0] and C11[1,1]
    U[0, col] = 1.0; U[3, col] = 1.0  # A[0,0], A[1,1]
    V[0, col] = 1.0; V[3, col] = 1.0  # B[0,0], B[1,1]
    W[0, col] = 1.0; W[4, col] = 1.0  # C[0,0], C[1,1]
    col += 1
    
    # M2 = (A11[1,0]+A11[1,1])*B11[0,0] -> contributes to C11[1,0]
    U[3, col] = 1.0; U[4, col] = 1.0  # A[1,0], A[1,1]
    V[0, col] = 1.0                    # B[0,0]
    W[3, col] = 1.0                    # C[1,0]
    col += 1
    
    # M3 = A11[0,0]*(B11[1,0]-B11[1,1]) -> contributes to C11[0,1] and C11[1,1]
    U[0, col] = 1.0                    # A[0,0]
    V[2, col] = 1.0; V[3, col] = -1.0  # B[1,0], -B[1,1]
    W[1, col] = 1.0; W[4, col] = 1.0   # C[0,1], C[1,1]
    col += 1
    
    # M4 = A11[1,1]*(B11[0,1]-B11[0,0]) -> contributes to C11[0,0] and C11[1,0]
    U[3, col] = 1.0                    # A[1,1]
    V[1, col] = 1.0; V[0, col] = -1.0  # B[0,1], -B[0,0]
    W[0, col] = 1.0; W[3, col] = 1.0   # C[0,0], C[1,0]
    col += 1
    
    # M5 = (A11[0,0]+A11[0,1])*B11[1,1] -> contributes to -C11[0,0] and C11[0,1]
    U[0, col] = 1.0; U[1, col] = 1.0  # A[0,0], A[0,1]
    V[3, col] = 1.0                    # B[1,1]
    W[0, col] = -1.0; W[1, col] = 1.0  # -C[0,0], C[0,1]
    col += 1
    
    # M6 = (A11[1,0]-A11[0,0])*(B11[0,0]+B11[0,1]) -> contributes to C11[1,1]
    U[3, col] = 1.0; U[0, col] = -1.0  # A[1,0], -A[0,0]
    V[0, col] = 1.0; V[1, col] = 1.0   # B[0,0], B[0,1]
    W[4, col] = 1.0                    # C[1,1]
    col += 1
    
    # M7 = (A11[0,1]-A11[1,1])*(B11[1,0]+B11[1,1]) -> contributes to C11[0,0]
    U[1, col] = 1.0; U[3, col] = -1.0  # A[0,1], -A[1,1]
    V[2, col] = 1.0; V[3, col] = 1.0   # B[1,0], B[1,1]
    W[0, col] = 1.0                    # C[0,0]
    col += 1
    
    # Cross terms: a1*b2 (A[0:2,2] * B[2,0:2] -> C[0:2,0:2])
    # A[0,2]*B[2,0] -> C[0,0]
    U[2, col] = 1.0
    V[6, col] = 1.0
    W[0, col] = 1.0
    col += 1
    # A[0,2]*B[2,1] -> C[0,1]
    U[2, col] = 1.0
    V[7, col] = 1.0
    W[1, col] = 1.0
    col += 1
    # A[1,2]*B[2,0] -> C[1,0]
    U[5, col] = 1.0
    V[6, col] = 1.0
    W[3, col] = 1.0
    col += 1
    # A[1,2]*B[2,1] -> C[1,1]
    U[5, col] = 1.0
    V[7, col] = 1.0
    W[4, col] = 1.0
    col += 1
    
    # Cross terms: A11*b1 (A[0:2,0:2] * B[0:2,2] -> C[0:2,2])
    # A[0,0]*B[0,2] -> C[0,2]
    U[0, col] = 1.0
    V[2, col] = 1.0
    W[2, col] = 1.0
    col += 1
    # A[0,1]*B[1,2] -> C[0,2]
    U[1, col] = 1.0
    V[5, col] = 1.0
    W[2, col] = 1.0
    col += 1
    # A[1,0]*B[0,2] -> C[1,2]
    U[3, col] = 1.0
    V[2, col] = 1.0
    W[5, col] = 1.0
    col += 1
    # A[1,1]*B[1,2] -> C[1,2]
    U[4, col] = 1.0
    V[5, col] = 1.0
    W[5, col] = 1.0
    col += 1
    
    # Cross terms: a1*b3 (A[0:2,2] * B[2,2] -> C[0:2,2])
    # A[0,2]*B[2,2] -> C[0,2]
    U[2, col] = 1.0
    V[8, col] = 1.0
    W[2, col] = 1.0
    col += 1
    # A[1,2]*B[2,2] -> C[1,2]
    U[5, col] = 1.0
    V[8, col] = 1.0
    W[5, col] = 1.0
    col += 1
    
    # Cross terms: a2*B11 (A[2,0:2] * B[0:2,0:2] -> C[2,0:2])
    # A[2,0]*B[0,0] -> C[2,0]
    U[6, col] = 1.0
    V[0, col] = 1.0
    W[6, col] = 1.0
    col += 1
    # A[2,0]*B[1,0] -> C[2,1]
    U[6, col] = 1.0
    V[3, col] = 1.0
    W[7, col] = 1.0
    col += 1
    # A[2,1]*B[0,0] -> C[2,0]
    U[7, col] = 1.0
    V[0, col] = 1.0
    W[6, col] = 1.0
    col += 1
    # A[2,1]*B[1,0] -> C[2,1]
    U[7, col] = 1.0
    V[3, col] = 1.0
    W[7, col] = 1.0
    col += 1
    
    # Cross terms: a3*b2 (A[2,2] * B[2,0:2] -> C[2,0:2])
    # A[2,2]*B[2,0] -> C[2,0]
    U[8, col] = 1.0
    V[6, col] = 1.0
    W[6, col] = 1.0
    col += 1
    # A[2,2]*B[2,1] -> C[2,1]
    U[8, col] = 1.0
    V[7, col] = 1.0
    W[7, col] = 1.0
    col += 1
    
    # Cross terms: a2*b1 (A[2,0:2] * B[0:2,2] -> C[2,2])
    # A[2,0]*B[0,2] -> C[2,2]
    U[6, col] = 1.0
    V[2, col] = 1.0
    W[8, col] = 1.0
    col += 1
    # A[2,1]*B[1,2] -> C[2,2]
    U[7, col] = 1.0
    V[5, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    # Cross terms: a3*b3 (A[2,2] * B[2,2] -> C[2,2])
    # A[2,2]*B[2,2] -> C[2,2]
    U[8, col] = 1.0
    V[8, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    return U, V, W


def _schoolbook():
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    # Try rank-26 decomposition first
    U, V, W = _rank26_decomposition()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
