"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a larger circulant structure.
    # We use the fact that for each fixed i, the inner product sum_j A[i,j]*B[j,k]
    # can be computed using a cyclic convolution if we embed appropriately.
    
    # Key insight: Use the DFT to diagonalize the circulant structure.
    # For the 5x7 part, we can use a cyclic convolution of size N where N >= m+p-1 = 11.
    # But we need integer/half-integer entries, so we use a careful construction.
    
    # Actually, let's use a simpler approach: the schoolbook gives rank 105.
    # We can improve by using the structure of the problem.
    
    # For matrix multiplication, the rank is at most n*m*p, but there are known
    # improvements. For small dimensions, we can use explicit constructions.
    
    # Let's use the following approach:
    # The tensor T[i,j,k] = 1 for i<n, j<m, k<p
    # We want to write T = sum_c U_c * V_c * W_c where each term is rank-1.
    
    # A simple improvement: use the fact that we can combine terms.
    # For each i, the terms over j,k form a matrix that we can decompose.
    
    # Let's use the Kronecker product structure:
    # T = sum_{i,j,k} e_i * e_j * e_k
    # where e_i is the i-th standard basis vector.
    
    # This gives rank = n*m*p = 105.
    
    # To improve, we can use the following trick:
    # For the 5x7 part, we can use a rank-reduced decomposition.
    # The 5x7 matrix multiplication tensor has rank at most 5*7 = 35.
    # But we can do better using structured decompositions.
    
    # For now, let's implement the schoolbook which is guaranteed correct,
    # and then try to find a better construction.
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
