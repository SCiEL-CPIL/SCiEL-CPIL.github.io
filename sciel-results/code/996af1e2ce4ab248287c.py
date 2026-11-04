import numpy as np

def construct():
    """
    Returns the schoolbook rank-27 decomposition for 3x3 matrix multiplication.
    
    The decomposition is T = sum_r U[:, r] V[:, r]^T W[:, r]^T
    where T is the matrix multiplication tensor.
    U, V, W have shapes (9, 27).
    
    Indices:
    A is 3x3, flattened to length 9 (row-major: i*3 + k)
    B is 3x3, flattened to length 9 (row-major: k*3 + j)
    C is 3x3, flattened to length 9 (row-major: i*3 + j)
    
    Wait, the task says:
    U: (n*m, rank) -> (3*3, rank) = (9, rank)
    V: (m*p, rank) -> (3*3, rank) = (9, rank)
    W: (p*n, rank) -> (3*3, rank) = (9, rank)
    
    Convention:
    A is n x m (3x3). Index (i, k) -> i*3 + k.
    B is m x p (3x3). Index (k, j) -> k*3 + j.
    C is n x p (3x3). Index (i, j) -> i*3 + j.
    
    The tensor T has indices (i, j, k) such that C_{ij} = sum_k A_{ik} B_{kj}.
    So T_{i j k} = 1.
    
    In the decomposition:
    U corresponds to A indices (i, k).
    V corresponds to B indices (k, j).
    W corresponds to C indices (i, j).
    
    For each term r:
    U[:, r] is a vector in R^9 (A space).
    V[:, r] is a vector in R^9 (B space).
    W[:, r] is a vector in R^9 (C space).
    
    The product is sum_r U_r[i,k] * V_r[k,j] * W_r[i,j].
    This must equal delta_{k, k} (which is 1) for the specific i,j,k?
    No, the tensor is T_{i j k}.
    The decomposition is T_{i j k} = sum_r U_r[i,k] V_r[k,j] W_r[i,j].
    
    Schoolbook:
    For each k in 0..2:
      For each i in 0..2:
        For each j in 0..2:
          Term r corresponds to (i, j, k).
          U_r[i, k] = 1 (A_{ik})
          V_r[k, j] = 1 (B_{kj})
          W_r[i, j] = 1 (C_{ij})
          All other entries 0.
    """
    n = 3
    m = 3
    p = 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # U index: i * m + k
                U[i * m + k, col] = 1.0
                # V index: k * p + j
                V[k * p + j, col] = 1.0
                # W index: i * p + j
                W[i * p + j, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
