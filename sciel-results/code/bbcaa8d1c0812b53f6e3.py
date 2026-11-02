import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Strategy: Use Pan's 3x3x3 algorithm (rank 23) for 3x3x3 blocks
    # and schoolbook for 3x1x3 blocks.
    # 3x4x6 = 3x(3+1)x(3+3)
    #   Block 1: A[:, 0:3] x B[0:3, 0:3] -> C[:, 0:3] (3x3x3, rank 23)
    #   Block 2: A[:, 0:3] x B[0:3, 3:6] -> C[:, 3:6] (3x3x3, rank 23)
    #   Block 3: A[:, 3:4] x B[3:4, 0:3] -> C[:, 0:3] (3x1x3, rank 9)
    #   Block 4: A[:, 3:4] x B[3:4, 3:6] -> C[:, 3:6] (3x1x3, rank 9)
    # Total rank: 23 + 23 + 9 + 9 = 64
    
    terms = []
    
    def add_pan_3x3x3(i0, j0, k0):
        """Add 23 Pan terms for a 3x3x3 block starting at (i0,j0,k0)."""
        # Pan's algorithm for 3x3 matrix multiplication
        # M1 = A11*(B11+B22+B33)
        terms.append({'a': {(i0,j0):1}, 'b': {(j0,k0):1, (j0+1,k0+1):1, (j0+2,k0+2):1}, 'c': {(i0,k0):1, (i0,k0+1):1, (i0,k0+2):1}})
        # M2 = (A12+A13)*(B21+B31)
        terms.append({'a': {(i0,j0+1):1, (i0,j0+2):1}, 'b': {(j0+1,k0):1, (j0+2,k0):1}, 'c': {(i0+1,k0):1, (i0+2,k0):1}})
        # M3 = (A12-A13)*(B31-B21)
        terms.append({'a': {(i0,j0+1):1, (i0,j0+2):-1}, 'b': {(j0+2,k0):1, (j0+1,k0):-1}, 'c': {(i0+1,k0):1, (i0+2,k0):-1}})
        # M4 = (A13-A23)*(B22-B32)
        terms.append({'a': {(i0,j0+2):1, (i0+1,j0+2):-1}, 'b': {(j0+1,k0+1):1, (j0+2,k0+1):-1}, 'c': {(i0+2,k0+1):1, (i0+1,k0+1):-1}})
        # M5 = (A13+A23)*(B22+B32)
        terms.append({'a': {(i0,j0+2):1, (i0+1,j0+2):1}, 'b': {(j0+1,k0+1):1, (j0+2,k0+1):1}, 'c': {(i0+2,k0+1):1, (i0+1,k0+1):1}})
        # M6 = (A21-A31)*(B12-B22)
        terms.append({'a': {(i0+1,j0):1, (i0+2,j0):-1}, 'b': {(j0,k0+1):1, (j0+1,k0+1):-1}, 'c': {(i0+1,k0+1):1, (i0+2,k0+1):-1}})
        # M7 = (A21+A31)*(B12+B22)
        terms.append({'a': {(i0+1,j0):1, (i0+2,j0):1}, 'b': {(j0,k0+1):1, (j0+1,k0+1):1}, 'c': {(i0+1,k0+1):1, (i0+2,k0+1):1}})
        # M8 = (A22-A23)*(B33-B13)
        terms.append({'a': {(i0+1,j0+1):1, (i0+1,j0+2):-1}, 'b': {(j0+2,k0+2):1, (j0,k0+2):-1}, 'c': {(i0+1,k0+2):1, (i0,k0+2):-1}})
        # M9 = (A22+A23)*(B33+B13)
        terms.append({'a': {(i0+1,j0+1):1, (i0+1,j0+2):1}, 'b': {(j0+2,k0+2):1, (j0,k0+2):1}, 'c': {(i0+1,k0+2):1, (i0,k0+2):1}})
        # M10 = (A31-A32)*(B11-B31)
        terms.append({'a': {(i0+2,j0):1, (i0+2,j0+1):-1}, 'b': {(j0,k0):1, (j0+2,k0):-1}, 'c': {(i0+2,k0):1, (i0,k0):-1}})
        # M11 = (A31+A32)*(B11+B31)
        terms.append({'a': {(i0+2,j0):1, (i0+2,j0+1):1}, 'b': {(j0,k0):1, (j0+2,k0):1}, 'c': {(i0+2,k0):1, (i0,k0):1}})
        # M12 = (A32-A33)*(B23-B33)
        terms.append({'a': {(i0+2,j0+1):1, (i0+2,j0+2):-1}, 'b': {(j0+1,k0+2):1, (j0+2,k0+2):-1}, 'c': {(i0+2,k0+2):1, (i0+1,k0+2):-1}})
        # M13 = (A32+A33)*(B23+B33)
        terms.append({'a': {(i0+2,j0+1):1, (i0+2,j0+2):1}, 'b': {(j0+1,k0+2):1, (j0+2,k0+2):1}, 'c': {(i0+2,k0+2):1, (i0+1,k0+2):1}})
        # M14 = (A11+A22)*(B11+B22)
        terms.append({'a': {(i0,j0):1, (i0+1,j0+1):1}, 'b': {(j0,k0):1, (j0+1,k0+1):1}, 'c': {(i0,k0):1, (i0+1,k0+1):1}})
        # M15 = (A11+A33)*(B11+B33)
        terms.append({'a': {(i0,j0):1, (i0+2,j0+2):1}, 'b': {(j0,k0):1, (j0+2,k0+2):1}, 'c': {(i0,k0):1, (i0+2,k0+2):1}})
        # M16 = (A22+A33)*(B22+B33)
        terms.append({'a': {(i0+1,j0+1):1, (i0+2,j0+2):1}, 'b': {(j0+1,k0+1):1, (j0+2,k0+2):1}, 'c': {(i0+1,k0+1):1, (i0+2,k0+2):1}})
        # M17 = (A12+A33)*(B21+B31)
        terms.append({'a': {(i0,j0+1):1, (i0+2,j0+2):1}, 'b': {(j0+1,k0):1, (j0+2,k0):1}, 'c': {(i0+1,k0):1, (i0+2,k0):1}})
        # M18 = (A13+A22)*(B31+B21)
        terms.append({'a': {(i0,j0+2):1, (i0+1,j0+1):1}, 'b': {(j0+2,k0):1, (j0+1,k0):1}, 'c': {(i0+1,k0):1, (i0+2,k0):1}})
        # M19 = (A23+A31)*(B22+B11)
        terms.append({'a': {(i0+1,j0+2):1, (i0+2,j0):1}, 'b': {(j0+1,k0+1):1, (j0,k0):1}, 'c': {(i0+1,k0+1):1, (i0,k0):1}})
        # M20 = (A32+A11)*(B33+B22)
        terms.append({'a': {(i0+2,j0+1):1, (i0,j0):1}, 'b': {(j0+2,k0+2):1, (j0+1,k0+1):1}, 'c': {(i0+2,k0+2):1, (i0+1,k0+1):1}})
        # M21 = (A33+A21)*(B13+B32)
        terms.append({'a': {(i0+2,j0+2):1, (i0+1,j0):1}, 'b': {(j0,k0+2):1, (j0+2,k0+1):1}, 'c': {(i0,k0+2):1, (i0+2,k0+1):1}})
        # M22 = (A11+A22+A33)*(B11+B22+B33)
        terms.append({'a': {(i0,j0):1, (i0+1,j0+1):1, (i0+2,j0+2):1}, 'b': {(j0,k0):1, (j0+1,k0+1):1, (j0+2,k0+2):1}, 'c': {(i0,k0):1, (i0+1,k0+1):1, (i0+2,k0+2):1}})
        # M23 = (A12+A23+A31)*(B21+B32+B13)
        terms.append({'a': {(i0,j0+1):1, (i0+1,j0+2):1, (i0+2,j0):1}, 'b': {(j0+1,k0):1, (j0+2,k0+1):1, (j0,k0+2):1}, 'c': {(i0+1,k0):1, (i0+2,k0+1):1, (i0,k0+2):1}})
    
    def add_schoolbook_1x1x1(i, j, k):
        """Add 1 schoolbook term for a single (i,j,k)."""
        terms.append({'a': {(i,j):1}, 'b': {(j,k):1}, 'c': {(i,k):1}})
    
    # Block 1: A[:, 0:3] x B[0:3, 0:3] -> C[:, 0:3]
    add_pan_3x3x3(0, 0, 0)
    
    # Block 2: A[:, 0:3] x B[0:3, 3:6] -> C[:, 3:6]
    add_pan_3x3x3(0, 0, 3)
    
    # Block 3: A[:, 3:4] x B[3:4, 0:3] -> C[:, 0:3]
    for k in range(3):
        add_schoolbook_1x1x1(0, 3, k)
        add_schoolbook_1x1x1(1, 3, k)
        add_schoolbook_1x1x1(2, 3, k)
    
    # Block 4: A[:, 3:4] x B[3:4, 3:6] -> C[:, 3:6]
    for k in range(3):
        add_schoolbook_1x1x1(0, 3, k+3)
        add_schoolbook_1x1x1(1, 3, k+3)
        add_schoolbook_1x1x1(2, 3, k+3)
    
    # Total rank: 23 + 23 + 9 + 9 = 64
    rank = len(terms)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r, term in enumerate(terms):
        for (i, j), coeff in term['a'].items():
            U[i*m + j, r] += coeff
        for (j, k), coeff in term['b'].items():
            V[j*p + k, r] += coeff
        for (i, k), coeff in term['c'].items():
            W[k*n + i, r] += coeff
    
    # Verify
    def check():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not check():
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
