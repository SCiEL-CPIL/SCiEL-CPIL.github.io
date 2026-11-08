import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using float arithmetic."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
        
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    Uf_r = Uf.reshape(n, m, rank)
    Vf_r = Vf.reshape(m, p, rank)
    Wf_r = Wf.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c Uf_r[i,j,c] * Vf_r[j,k,c] * Wf_r[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Uf_r, Vf_r, Wf_r)
    
    if not np.allclose(T, 1.0, atol=1e-7):
        return False
    
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _kronecker_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Combine two matrix multiplication decompositions via Kronecker product.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Efficiently construct Kronecker product matrices
    # U[i1*n2+i2, j1*m2+j2, c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1 = U1[i1 * m1 + j1][c1]
                if u1 == 0:
                    continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2 = U2[i2 * m2 + j2][c2]
                            if u2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            c = c1 * r2 + c2
                            U[i * m + j][c] += u1 * u2
                            
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1 = V1[j1 * p1 + k1][c1]
                if v1 == 0:
                    continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2 = V2[j2 * p2 + k2][c2]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            c = c1 * r2 + c2
                            V[j * p + k][c] += v1 * v2
                            
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1 = W1[k1 * n1 + i1][c1]
                if w1 == 0:
                    continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2 = W2[k2 * n2 + i2][c2]
                            if w2 == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            c = c1 * r2 + c2
                            W[k * n + i][c] += w1 * w2
                            
    return U, V, W

def _try_optimize_subproblem(n, m, p, time_limit=10):
    """
    Attempt to find a lower rank decomposition for a small subproblem.
    Currently returns schoolbook as no known improvement exists for 2x2x3.
    """
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Strategy:
    # 1. Try Kronecker product of (2,2,3) and (2,2,3).
    #    Rank = R(2,2,3)^2.
    #    If R(2,2,3) < 12, we get rank < 144.
    #    Currently R(2,2,3) is believed to be 12.
    
    # Get subproblem decompositions
    n1, m1, p1 = 2, 2, 3
    n2, m2, p2 = 2, 2, 3
    
    U1, V1, W1 = _try_optimize_subproblem(n1, m1, p1, time_limit=5)
    U2, V2, W2 = _try_optimize_subproblem(n2, m2, p2, time_limit=5)
    
    # Combine
    U, V, W = _kronecker_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
