"""Rewritten program: composition-based decomposition for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.
    
    The approach uses the schoolbook decomposition directly, which gives rank n*m*p = 36.
    However, we can potentially do better by exploiting structure.
    
    For 2x3x6 matrix multiplication, the standard lower bound is max(n*m, m*p, n*p) = max(6, 18, 12) = 18.
    The schoolbook rank is 2*3*6 = 36.
    
    We use the schoolbook decomposition as a baseline. The tensor for multiplying (n x m) by (m x p)
    has entries T[i, j, k] = sum_l A[i,l] * B[l,k]. The rank of this tensor is at most n*m*p.
    
    For 2x3x6, the schoolbook rank is 36. Let's construct it properly.
    
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (18, rank)
    W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = delta (i.e., the multiplication tensor has 1s when the indices align properly).
    
    Actually, the matrix multiplication tensor T_{i,j,k} for multiplying (n x m) by (m x p) is:
    T_{i,j,k} = sum_l e_{i,l} e_{l,k} where e_{i,l} is from A and e_{l,k} is from B.
    
    The standard schoolbook decomposition: for each (i, l, k) triple, we have a rank-1 term
    where U[i*m+l, c] = 1, V[l*p+k, c] = 1, W[k*n+i, c] = 1, and all other entries in column c are 0.
    
    This gives rank = n*m*p = 2*3*6 = 36.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U[i * m + l][column] = 1
                V[l * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
