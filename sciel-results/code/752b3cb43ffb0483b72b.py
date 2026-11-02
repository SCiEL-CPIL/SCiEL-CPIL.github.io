import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Build decomposition using block splitting with Strassen for 2x2x2 blocks
    # Strategy: recursively split dimensions, use Strassen when possible
    
    # We'll build a list of terms. Each term is a dict with:
    # 'a': {(i,j): coeff} - coefficients for A[i,j]
    # 'b': {(j,k): coeff} - coefficients for B[j,k]
    # 'c': {(i,k): coeff} - coefficients for C[i,k]
    
    terms = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 block starting at (i0,j0,k0)."""
        # A block: rows i0,i0+1; cols j0,j0+1
        # B block: rows j0,j0+1; cols k0,k0+1
        # C block: rows i0,i0+1; cols k0,k0+1
        
        # M1 = (A11+A22)(B11+B22)
        terms.append({
            'a': {(i0,j0):1, (i0+1,j0+1):1},
            'b': {(j0,k0):1, (j0+1,k0+1):1},
            'c': {(i0,k0):1, (i0+1,k0+1):1}
        })
        # M2 = (A21+A22)B11
        terms.append({
            'a': {(i0+1,j0):1, (i0+1,j0+1):1},
            'b': {(j0,k0):1},
            'c': {(i0+1,k0):1, (i0+1,k0+1):-1}
        })
        # M3 = A11(B12-B22)
        terms.append({
            'a': {(i0,j0):1},
            'b': {(j0,k0+1):1, (j0+1,k0+1):-1},
            'c': {(i0,k0+1):1, (i0+1,k0+1):1}
        })
        # M4 = A22(B21-B11)
        terms.append({
            'a': {(i0+1,j0+1):1},
            'b': {(j0+1,k0):1, (j0,k0):-1},
            'c': {(i0,k0):1, (i0+1,k0):1}
        })
        # M5 = (A11+A12)B22
        terms.append({
            'a': {(i0,j0):1, (i0,j0+1):1},
            'b': {(j0+1,k0+1):1},
            'c': {(i0,k0):-1, (i0,k0+1):1}
        })
        # M6 = (A21-A11)(B11+B12)
        terms.append({
            'a': {(i0+1,j0):1, (i0,j0):-1},
            'b': {(j0,k0):1, (j0,k0+1):1},
            'c': {(i0+1,k0+1):1}
        })
        # M7 = (A12-A22)(B21+B22)
        terms.append({
            'a': {(i0,j0+1):1, (i0+1,j0+1):-1},
            'b': {(j0+1,k0):1, (j0+1,k0+1):1},
            'c': {(i0,k0):1}
        })
    
    def add_schoolbook_1x1x1(i, j, k):
        """Add 1 schoolbook term for a single (i,j,k)."""
        terms.append({
            'a': {(i,j):1},
            'b': {(j,k):1},
            'c': {(i,k):1}
        })
    
    # Decompose 3x4x6:
    # Split rows of A into [0:2] and [2:3]
    # Split middle dim into [0:2] and [2:4]
    # Split cols of B into [0:6]
    
    # For A[0:2, 0:2] x B[0:2, 0:6]:
    #   This is 2x2 x 2x6. Split B cols into [0:2], [2:4], [4:6]
    #   For each 2x2 x 2x2 block, use Strassen
    add_strassen_2x2x2(0, 0, 0)  # A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 0, 2)  # A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_2x2x2(0, 0, 4)  # A[0:2,0:2] x B[0:2,4:6] -> C[0:2,4:6]
    
    # For A[0:2, 2:4] x B[2:4, 0:6]:
    add_strassen_2x2x2(0, 2, 0)  # A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 2, 2)  # A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]
    add_strassen_2x2x2(0, 2, 4)  # A[0:2,2:4] x B[2:4,4:6] -> C[0:2,4:6]
    
    # For A[2:3, 0:2] x B[0:2, 0:6]:
    # 1x2 x 2x6: use schoolbook for each (i,j,k)
    for k in range(6):
        add_schoolbook_1x1x1(2, 0, k)
        add_schoolbook_1x1x1(2, 1, k)
    
    # For A[2:3, 2:4] x B[2:4, 0:6]:
    for k in range(6):
        add_schoolbook_1x1x1(2, 2, k)
        add_schoolbook_1x1x1(2, 3, k)
    
    # Total rank: 6 Strassen blocks * 7 + 4 schoolbook * 6 = 42 + 24 = 66
    
    # Now build U, V, W matrices
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
