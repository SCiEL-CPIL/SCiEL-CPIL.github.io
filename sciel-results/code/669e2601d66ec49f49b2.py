"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to compress via SVD
    # Concatenate U, V, W vertically to find joint structure
    # Actually, the tensor is T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We want to find a lower-rank representation.
    # One approach: view the tensor as a matrix and do SVD.
    
    # Reshape tensor: T[i, j, k] -> matrix of size (n*m) x p or (n) x (m*p)
    # Let's use the matricization: M[i, (j,k)] = T[i,j,k]
    # This is n x (m*p) = 4 x 24
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # This is wrong, the tensor entries are all 1? No.
    
    # Actually the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in the standard basis? 
    # No. The tensor is defined by T(A,B) = A*B. In tensor form, T[i,j,k] is the coefficient 
    # of A[i,j]*B[j,k] in the output (A*B)[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # So the tensor is all-ones! That's why schoolbook works with all 1s.
    
    # So T = ones(n,m,p). We need to decompose this as sum of rank-1 tensors.
    # The minimal rank of the all-ones tensor of size 4x4x6 is what we're looking for.
    
    # Let's use SVD on the matricization to get a lower-rank approximation, but we need EXACT.
    # Since T is all-ones, it has rank 1 in any matricization! But that's for the matrix, not the tensor.
    # The tensor rank is different.
    
    # For exact decomposition, let's just use schoolbook and return it.
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
