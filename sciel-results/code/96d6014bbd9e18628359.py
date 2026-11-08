"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal (schoolbook) exact tensor decomposition 
    for multiplying a 2x5 matrix by a 5x5 matrix.
    
    The decomposition is of the form:
    T = sum_{c=0}^{rank-1} U_c \otimes V_c \otimes W_c
    
    Where:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The multiplication tensor T is defined such that:
    T[i, j, k, l, m, n] = 1 if i==l and j==m and k==n (indices adjusted for flattening)
    
    Specifically for C = A B:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor notation:
    T_{i,j,k}^{C} = sum_{j} A_{i,j} B_{j,k}
    
    The factorization U, V, W satisfies:
    T[i, j, k] = sum_{c} U[i, c] V[j, c] W[k, c]  (This is a simplified view)
    
    More precisely, the task defines:
    U: (n*m, rank) -> U[i*m + j, c]
    V: (m*p, rank) -> V[j*p + k, c]
    W: (p*n, rank) -> W[k*n + i, c]
    
    And the condition is:
    sum_{c} U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1 
    if this corresponds to a valid multiplication term, else 0.
    
    Actually, the standard tensor for matmul (n x m) * (m x p) is T[i, j, k] where:
    i in [0, n), j in [0, m), k in [0, p).
    T[i, j, k] = 1 if the indices match the multiplication flow?
    
    The task description says:
    "exactly decompose the matrix multiplication tensor"
    "output factor follows the official trace convention W[k*n+i,r]"
    
    The schoolbook decomposition sets:
    For each (i, j, k):
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    This ensures that the term A[i,j] * B[j,k] contributes to C[i,k].
    The index for W is k*n + i.
    This matches the convention W[k*n+i, r].
    
    Rank = n * m * p = 2 * 5 * 5 = 50.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the specific entries for this term
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
