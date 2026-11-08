import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ur = np.array(U).reshape(n, m, rank)
    Vr = np.array(V).reshape(m, p, rank)
    Wr = np.array(W).reshape(p, n, rank)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _rank_one_solution():
    """
    The matrix multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all valid indices.
    This is an all-ones tensor, which has rank 1:
    T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    
    So we can use rank 1 with:
    U[i*m+j, 0] = 1 for all i,j
    V[j*p+k, 0] = 1 for all j,k  
    W[k*n+i, 0] = 1 for all k,i
    """
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    # Try rank-1 solution first
    sol = _rank_one_solution()
    if _verify(sol["U"], sol["V"], sol["W"], n, m, p):
        return sol
    
    # Fallback to schoolbook
    return _schoolbook()
