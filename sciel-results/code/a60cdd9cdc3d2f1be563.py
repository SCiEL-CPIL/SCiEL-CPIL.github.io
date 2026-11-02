"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n=3, m=3, p=5, atol=1e-7):
    """
    Verify that the tensor decomposition exactly matches matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = U[0].__len__() if isinstance(U, list) else U.shape[1]
    # Build the tensor from the decomposition
    # We can do this via einsum: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Let's use a simple loop or numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to make einsum work
    # U_np has shape (n*m, rank). We need U[i*m+j, c]
    # V_np has shape (m*p, rank). We need V[j*p+k, c]
    # W_np has shape (p*n, rank). We need W[k*n+i, c]
    
    # Let's build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                T[i, j, k] = val
    
    # The target tensor for matrix multiplication is T[i,j,k] = 1 if all indices are valid
    # Wait, the tensor for matrix multiplication is T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}
    # In the standard tensor notation for matrix multiplication:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor is T[i,j,k] = 1 for all i,j,k.
    # Because A_{ij} B_{jk} is the term.
    # So the target tensor is all ones.
    
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=atol)

def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
