"""Program for matrix multiplication tensor decomposition (2,4,4)."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The tensor T[i,j,k] = 1 for all i,j,k (in the sense that A[i,j]*B[j,k] contributes
    to C[i,k] with coefficient 1).
    
    We use a rank-16 construction based on the following insight:
    For each output column k (0..3), we need to compute C[i,k] = sum_j A[i,j] B[j,k].
    For fixed k, this is a bilinear form in (A[:,k_col], B[k_row,:]).
    
    The key observation: for fixed k, we can write
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Using 4 terms per k (one for each j), we get:
    Term (k, j): U[i*m+j, k*m+j] * V[j*p+k, k*m+j] * W[k*n+i, k*m+j]
    
    For this term:
    - U[i*m+j, k*m+j] = 1 if the row index matches i*m+j (always true for our i)
    - V[j*p+k, k*m+j] = 1 if the row index matches j*p+k (always true for our j)
    - W[k*n+i, k*m+j] = 1
    
    Wait, this doesn't work directly. Let me reconsider.
    
    Actually, the schoolbook uses 32 terms: one for each (i,j,k) triple.
    
    A better approach: For each k, the computation C[i,k] = sum_j A[i,j] B[j,k]
    can be seen as: for each j, we have a product A[i,j] * B[j,k].
    
    We can share the W factor across i values for the same (j,k).
    
    Construction with rank 16:
    For each (j,k) pair (16 total), create one term:
    - U[i*m+j, j*p+k] = 1 for all i (so U[4*i+j, 4*j+k] = 1)
    - V[j*p+k, j*p+k] = 1
    - W[k*n+i, j*p+k] = 1 for all i (so W[2*k+i, 4*j+k] = 1)
    
    Check: sum over terms (j',k') of U[i*m+j, j'*p+k'] * V[j*p+k, j'*p+k'] * W[k*n+i, j'*p+k']
    = sum over (j',k') of [1 if j==j' else 0] * [1 if (j,k)==(j',k') else 0] * [1 if k==k' else 0]
    Wait, this isn't right either.
    
    Let me think more carefully. We need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Try rank 16 with terms indexed by (j,k):
    U[i*m+j, j*p+k] = 1 for all i
    V[j*p+k, j*p+k] = 1
    W[k*n+i, j*p+k] = 1 for all i
    
    Then for a given (i,j,k):
    sum over (j',k') of U[i*m+j, j'*p+k'] * V[j*p+k, j'*p+k'] * W[k*n+i, j'*p+k']
    
    U[i*m+j, j'*p+k'] = 1 iff j == j'
    V[j*p+k, j'*p+k'] = 1 iff (j,k) == (j',k')
    W[k*n+i, j'*p+k'] = 1 iff k == k'
    
    So the product is 1 iff j==j' AND (j,k)==(j',k') AND k==k', which simplifies to (j,k)==(j',k').
    The sum is 1. 
    
    This gives rank 16!
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = m * p  # 16
    
    # U has shape (n*m, rank) = (8, 16)
    # V has shape (m*p, rank) = (16, 16)
    # W has shape (p*n, rank) = (8, 16)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            term_idx = j * p + k
            
            # U[i*m + j, term_idx] = 1 for all i
            for i in range(n):
                U[i * m + j][term_idx] = 1
            
            # V[j*p + k, term_idx] = 1
            V[j * p + k][term_idx] = 1
            
            # W[k*n + i, term_idx] = 1 for all i
            for i in range(n):
                W[k * n + i][term_idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
