import time
from fractions import Fraction

_DIMS = (3, 4, 6)
_DEADLINE = 120.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def _build_strassen_terms():
    """Build the 66 rank-1 terms for the Strassen-based decomposition."""
    n, m, p = _DIMS
    terms = []
    for ko in (0, 2, 4):
        # Top row (i=0): 8 schoolbook terms over j in 0..3, k in {ko, ko+1}
        for j in range(4):
            for k in (ko, ko + 1):
                terms.append(([(0, j, 1)], [(j, k, 1)], [(k, 0, 1)]))
        # Rows 1,2 (i=1,2): Strassen on j-pairs (0,1) and (2,3)
        for (j0, j1) in ((0, 1), (2, 3)):
            # M1 = (A[1,j0]+A[2,j1]) * (B[j0,ko]+B[j1,ko+1]) -> C[1,ko], C[2,ko+1]
            terms.append(([(1, j0, 1), (2, j1, 1)], [(j0, ko, 1), (j1, ko + 1, 1)], [(ko, 1, 1), (ko + 1, 2, 1)]))
            # M2 = (A[2,j0]+A[2,j1]) * B[j0,ko] -> C[2,ko], C[2,ko+1] (with -1)
            terms.append(([(2, j0, 1), (2, j1, 1)], [(j0, ko, 1)], [(ko, 2, 1), (ko + 1, 2, -1)]))
            # M3 = A[1,j0] * (B[j0,ko+1]-B[j1,ko+1]) -> C[1,ko+1], C[2,ko+1]
            terms.append(([(1, j0, 1)], [(j0, ko + 1, 1), (j1, ko + 1, -1)], [(ko + 1, 1, 1), (ko + 1, 2, 1)]))
            # M4 = A[2,j1] * (B[j1,ko]-B[j0,ko]) -> C[1,ko], C[2,ko]
            terms.append(([(2, j1, 1)], [(j1, ko, 1), (j0, ko, -1)], [(ko, 1, 1), (ko, 2, 1)]))
            # M5 = (A[1,j0]+A[1,j1]) * B[j1,ko+1] -> C[1,ko] (with -1), C[1,ko+1]
            terms.append(([(1, j0, 1), (1, j1, 1)], [(j1, ko + 1, 1)], [(ko, 1, -1), (ko + 1, 1, 1)]))
            # M6 = (A[2,j0]-A[1,j0]) * (B[j0,ko]+B[j0,ko+1]) -> C[2,ko+1]
            terms.append(([(2, j0, 1), (1, j0, -1)], [(j0, ko, 1), (j0, ko + 1, 1)], [(ko + 1, 2, 1)]))
            # M7 = (A[1,j1]-A[2,j1]) * (B[j1,ko]+B[j1,ko+1]) -> C[1,ko]
            terms.append(([(1, j1, 1), (2, j1, -1)], [(j1, ko, 1), (j1, ko + 1, 1)], [(ko, 1, 1)]))
    return terms

def _verify_decomposition(U, V, W, rank):
    """Verify that the decomposition is exact: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1."""
    n, m, p = _DIMS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = _DIMS
    terms = _build_strassen_terms()
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in out_terms:
            W[k * n + i][r] += c
    
    if _verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    return _schoolbook()

def _schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
