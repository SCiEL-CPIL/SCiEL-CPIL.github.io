import time

_DIMS = (3, 4, 6)
_DEADLINE = 120.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)


def _build_decomposition():
    """Build rank-66 decomposition: 3 blocks of (8 schoolbook + 14 Strassen)."""
    n, m, p = _DIMS
    terms = []  # Each term: (a_list, b_list, w_list)
    # a_list: [(i, j, coeff), ...]
    # b_list: [(j, k, coeff), ...]
    # w_list: [(k, i, coeff), ...]

    for ko in (0, 2, 4):
        # Row i=0: schoolbook 4*2 = 8 terms
        for j in range(4):
            for k in (ko, ko + 1):
                terms.append(([(0, j, 1)], [(j, k, 1)], [(k, 0, 1)]))

        # Rows i=1,2: Strassen on each (j0,j1) pair in {(0,1), (2,3)}
        for (j0, j1) in ((0, 1), (2, 3)):
            # M1: (A[1,j0]+A[2,j1])*(B[j0,ko]+B[j1,ko+1]) -> C[1,ko]+C[2,ko+1]
            terms.append((
                [(1, j0, 1), (2, j1, 1)],
                [(j0, ko, 1), (j1, ko + 1, 1)],
                [(ko, 1, 1), (ko + 1, 2, 1)]
            ))
            # M2: (A[2,j0]+A[2,j1])*B[j0,ko] -> C[2,ko]-C[2,ko+1]
            terms.append((
                [(2, j0, 1), (2, j1, 1)],
                [(j0, ko, 1)],
                [(ko, 2, 1), (ko + 1, 2, -1)]
            ))
            # M3: A[1,j0]*(B[j0,ko+1]-B[j1,ko+1]) -> C[1,ko+1]+C[2,ko+1]
            terms.append((
                [(1, j0, 1)],
                [(j0, ko + 1, 1), (j1, ko + 1, -1)],
                [(ko + 1, 1, 1), (ko + 1, 2, 1)]
            ))
            # M4: A[2,j1]*(B[j1,ko]-B[j0,ko]) -> C[1,ko]+C[2,ko]
            terms.append((
                [(2, j1, 1)],
                [(j1, ko, 1), (j0, ko, -1)],
                [(ko, 1, 1), (ko, 2, 1)]
            ))
            # M5: (A[1,j0]+A[1,j1])*B[j1,ko+1] -> -C[1,ko]+C[1,ko+1]
            terms.append((
                [(1, j0, 1), (1, j1, 1)],
                [(j1, ko + 1, 1)],
                [(ko, 1, -1), (ko + 1, 1, 1)]
            ))
            # M6: (A[2,j0]-A[1,j0])*(B[j0,ko]+B[j0,ko+1]) -> C[2,ko+1]
            terms.append((
                [(2, j0, 1), (1, j0, -1)],
                [(j0, ko, 1), (j0, ko + 1, 1)],
                [(ko + 1, 2, 1)]
            ))
            # M7: (A[1,j1]-A[2,j1])*(B[j1,ko]+B[j1,ko+1]) -> C[1,ko]
            terms.append((
                [(1, j1, 1), (2, j1, -1)],
                [(j1, ko, 1), (j1, ko + 1, 1)],
                [(ko, 1, 1)]
            ))

    return terms


def _verify(U, V, W, rank):
    """Verify tensor identity: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta[i,j,k]... 
    Actually: C[i,k] = sum_j A[i,j]*B[j,k]. The tensor is T[i,j,k]=1 for all (i,j,k).
    Decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]... 
    
    Wait, the convention is: W[k*n+i, r]. So W is indexed by (k,i) where k in 0..p-1, i in 0..n-1.
    The tensor equation is: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
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
    terms = _build_decomposition()
    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (a_list, b_list, w_list) in enumerate(terms):
        for (i, j, c) in a_list:
            U[i * m + j][r] += c
        for (j, k, c) in b_list:
            V[j * p + k][r] += c
        for (k, i, c) in w_list:
            W[k * n + i][r] += c

    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
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
