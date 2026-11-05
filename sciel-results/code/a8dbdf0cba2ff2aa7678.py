"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 5, 7
    # Schoolbook decomposition: rank = 3*5*7 = 105
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
    
    # Try to find a lower rank decomposition using random search
    # The tensor is T_{i,j,k} = 1 for all i,j,k
    # We want to find rank r < 105 such that T = sum_{t=1}^{r} U_t * V_t * W_t
    
    # For this specific tensor (all ones), we can try to find a low-rank decomposition
    # The tensor is rank 1! Because T_{i,j,k} = 1 = 1*1*1 for all i,j,k
    # Wait, that's not right. The matrix multiplication tensor is different.
    
    # The matrix multiplication tensor T_{(i,j),(j,k),(k,i)} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually, for matrix multiplication, the tensor is:
    # T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Let me think about this more carefully.
    # The bilinear form for matrix multiplication C = AB is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # The tensor T has indices (i,j), (j,k), (i,k) and T_{(i,j),(j,k),(i,k)} = 1
    # This is the matrix multiplication tensor.
    
    # For the all-ones case, the tensor is actually the matrix multiplication tensor,
    # not the all-ones tensor.
    
    # Let me try a different approach: use the known lower bound and try to find
    # a decomposition with rank slightly above the lower bound.
    
    # The Strassen algorithm gives rank 7 for 2x2x2.
    # For 3x5x7, we can try to use a Kronecker product approach.
    
    # Actually, let me just try to use numpy to find a low-rank decomposition
    # by solving a linear system.
    
    # The tensor T can be flattened to a matrix of size (n*m) x (m*p) for each output index.
    # Or we can flatten it to a vector of length n*m*p.
    
    # Let me try a random search approach:
    # Generate random rank-r decompositions and check if they match the tensor.
    
    # For the all-ones matrix multiplication tensor, the rank is at most 105 (schoolbook).
    # Let me try to find a lower rank.
    
    # Actually, I realize the tensor for matrix multiplication is:
    # T_{(i,j),(j,k),(k,i)} = 1 if the indices match appropriately
    # Specifically, T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Let me construct the tensor properly and try to find its rank.
    
    # Flatten the tensor to a matrix: T is a 3D array of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    # To find the rank, we can flatten T to a matrix of shape (n*m) x (m*p*p*n)
    # and compute its rank.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Flatten to matrix: (n*m) x (m*p * p*n)
    T_mat = T.reshape(n * m, m * p * p * n)
    rank_T = np.linalg.matrix_rank(T_mat)
    
    # The rank of the tensor is the minimum r such that T can be written as sum of r rank-1 tensors.
    # This is the tensor rank, which is different from the matrix rank.
    
    # Let me try a different approach: use the fact that the matrix multiplication tensor
    # for 3x5x7 might have a known rank.
    
    # For now, let me just return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
