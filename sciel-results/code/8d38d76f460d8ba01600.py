import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Identity matrices for the three dimensions
    I_n = np.eye(n)
    I_m = np.eye(m)
    I_p = np.eye(p)
    
    # We want to form U, V, W such that U @ diag(1s) @ V^T * ... 
    # Actually, the schoolbook decomposition is a sum of rank 1 tensors.
    # Tensor T_{ijk} = delta_{j, j} delta_{k, k} ... 
    # T_{i,j,k} = 1 if indices match in the product? No.
    # The tensor for multiplication is T_{i, j, k; i', j', k'} ... no.
    # The tensor is indexed by (row of A, col of A, row of B, col of B) -> (row of C, col of C).
    # Standard form: U (n*m x r), V (m*p x r), W (p*n x r).
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = C[i, k]
    
    # Schoolbook: c = (i, j, k)
    # U[i*m+j, (i,j,k)] = 1
    # V[j*p+k, (i,j,k)] = 1
    # W[k*n+i, (i,j,k)] = 1
    
    # Using Kronecker products:
    # U = I_n * I_m * I_p? 
    # Let's build the index mapping.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
