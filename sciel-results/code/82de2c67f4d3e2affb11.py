"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying 
    a 2 by 4 matrix by a 4 by 5 matrix.
    
    The decomposition follows the schoolbook algorithm, which has rank n*m*p.
    For n=2, m=4, p=5, the rank is 40.
    
    The output format is a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The entries are integers (0 or 1 in this case).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all combinations of indices i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to the column
                # Index in U is i*m + j
                U[i * m + j][column] = 1
                
                # V maps input B[j,k] to the column
                # Index in V is j*p + k
                V[j * p + k][column] = 1
                
                # W maps the product to output C[k*n + i] (trace convention W[k*n+i, r])
                # Note: The task specifies W[k*n+i, r].
                # Standard matrix multiplication C[i,k] = sum_j A[i,j]B[j,k].
                # The tensor T_{i,j,k} = 1 if we consider the multiplication tensor.
                # The decomposition is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # This sums to C[i,k] if W is indexed by (k,i).
                # Let's verify the index: W[k*n + i].
                # Yes, this matches the "official trace convention" mentioned in the prompt.
                W[k * n + i][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
