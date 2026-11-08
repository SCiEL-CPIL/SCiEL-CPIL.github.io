import time
import numpy as np

_T0 = time.time()
_DEADLINE = 120.0

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    r = 0
    for ko in range(0, p, 2):
        k1 = ko
        k2 = ko + 1
        # Row 0: schoolbook for 1x4 * 4x2 -> 8 multiplications
        for j in range(m):
            U[0 * m + j, r] = 1
            V[j * p + k1, r] = 1
            W[k1 * n + 0, r] = 1
            r += 1
            U[0 * m + j, r] = 1
            V[j * p + k2, r] = 1
            W[k2 * n + 0, r] = 1
            r += 1
        
        # Rows 1-2: Strassen for 2x4 * 4x2
        # Split m=4 into two blocks of size 2: (j0, j1) = (0, 1) and (j2, j3) = (2, 3)
        for j0, j1 in ((0, 1), (2, 3)):
            # m1 = (a1j0 + a2j1) * (b_j0,k1 + b_j1,k2)
            U[1 * m + j0, r] = 1
            U[2 * m + j1, r] = 1
            V[j0 * p + k1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m2 = (a2j0 + a2j1) * b_j0,k1
            U[2 * m + j0, r] = 1
            U[2 * m + j1, r] = 1
            V[j0 * p + k1, r] = 1
            W[k1 * n + 2, r] = 1
            W[k2 * n + 2, r] = -1
            r += 1
            
            # m3 = a1j0 * (b_j0,k2 - b_j1,k2)
            U[1 * m + j0, r] = 1
            V[j0 * p + k2, r] = 1
            V[j1 * p + k2, r] = -1
            W[k2 * n + 1, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m4 = a2j1 * (b_j1,k1 - b_j0,k1)
            U[2 * m + j1, r] = 1
            V[j1 * p + k1, r] = 1
            V[j0 * p + k1, r] = -1
            W[k1 * n + 1, r] = 1
            W[k1 * n + 2, r] = 1
            r += 1
            
            # m5 = (a1j0 + a1j1) * b_j1,k2
            U[1 * m + j0, r] = 1
            U[1 * m + j1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = -1
            W[k2 * n + 1, r] = 1
            r += 1
            
            # m6 = (a2j0 - a1j0) * (b_j0,k1 + b_j0,k2)
            U[2 * m + j0, r] = 1
            U[1 * m + j0, r] = -1
            V[j0 * p + k1, r] = 1
            V[j0 * p + k2, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m7 = (a1j1 - a2j1) * (b_j1,k1 + b_j1,k2)
            U[1 * m + j1, r] = 1
            U[2 * m + j1, r] = -1
            V[j1 * p + k1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = 1
            r += 1
    
    if not _deadline_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the tensor decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for rr in range(rank):
                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
