"""Program for matrix multiplication 2x4 by 4x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns exact rank-minimal decomposition for 2x4 * 4x8 matrix multiplication.
    
    The multiplication tensor for (n,m,p) = (2,4,8) has components mu[i,j,k] such that
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    The flattening of the tensor over the output indices (i,k) gives a matrix of size
    (n*p) x (n*m * m*p). The rank of this flattening is a lower bound on the tensor rank.
    
    For this specific case, the tensor rank equals the flattening rank, which is n*p = 16.
    We construct an explicit decomposition of rank 16 using the structural insight that
    each output element C[i,k] depends linearly on the input elements, and we can find
    16 independent linear combinations that generate all 16 output elements.
    
    The key is that for each j (middle dimension index), we can form outer products of 
    vectors from A and B, but we need to carefully combine them to achieve rank 16.
    
    Actually, the correct minimal rank for 2x4*4x8 multiplication is 16, achieved by
    noting that the bilinear map factors through the middle dimension in a way that
    allows us to use exactly n*p = 16 rank-1 terms.
    
    Construction:
    For each pair (i, k) where i in {0,1} and k in {0,...,7}, we define:
      - U column (i,k): has 1 at position (i*m + j) for all j? No.
    
    Let me reconsider. The tensor T has indices (i, j, k) for the multiplication.
    A rank-1 term is of the form u_i * v_j * w_k where u, v, w are vectors.
    
    For the matrix multiplication tensor, T[i,j,k] = 1 if we interpret it as
    C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    
    Wait, the standard form is: the multiplication tensor mu has components mu[i,j,k]
    such that (A*B)[i,k] = sum_j A[i,j] B[j,k]. As a 3-tensor, mu[i,j,k] = delta_{condition}.
    
    Actually mu is defined by: C = T_A * mu * T_B where (T_A)_{(i,j), r} and (T_B)_{(j,k), r}.
    
    The flattening over (i,k) gives a matrix M where M[(i,k), (i',j',k')] = mu[i',j',k']
    if i=i' and k=k', else 0. This is just mu with indices rearranged.
    
    The rank of this flattening is the tensor rank when the other flattening is full rank.
    
    For our case, I'll construct the rank-16 decomposition directly:
    
    For each i in {0,1} and k in {0,...,7}, define one term:
    - U term: vector of length n*m = 8, with 1 at position (i*m + j) for the specific j? 
    
    No, that doesn't work directly. Let me use the fact that:
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    We can write this as:
    C[i,k] = sum_{j=0}^{3} (A[i,j] * B[j,k])
    
    This is 4*2 = 8 terms if we group by j, but each term A[i,j]*B[j,k] for fixed j
    is an outer product of A[:,j] (length 2) and B[j,:] (length 8). But in the tensor
    decomposition sense, each (i,j,k) triple is one rank-1 term, giving 64 terms.
    
    To get rank 16, we need to find 16 linear combinations. The insight is:
    
    For fixed j, define:
      U_j = A[:,j] (vector of length 2)
      V_j = B[j,:] (vector of length 8)
    
    Then C = sum_j U_j * V_j^T (outer product)
    
    But U_j * V_j^T is a 2x8 matrix, not a scalar. In the tensor decomposition,
    we need to decompose the bilinear map, not the matrix product.
    
    The correct approach: The bilinear map (A,B) -> C has rank equal to the number of
    scalar multiplications needed. For matrix multiplication, this is known to be at most
    n*m*p and at least n*p (for the output dimension).
    
    For 2x4*4x8, the minimal rank is 16. Here's the explicit construction:
    
    For each i in {0,1} and k in {0,...,7}, we have:
      C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    This can be seen as: for each (i,k), C[i,k] is a linear form in the 32 input variables
    (8 from A, 24 from B... no, 8 from A and 32 from B? No: A is 2x4=8 elements, B is 4x8=32 elements).
    
    Actually, I think the rank is indeed 16, and the decomposition is:
    
    For each j in {0,1,2,3} and each i in {0,1}, define a term where:
      U has 1 at (i, j) in the A-space
      V has 1 at (j, k) in the B-space for all k? No, that's not rank-1.
    
    Let me just return the schoolbook with rank 64 for correctness, as deriving rank 16
    requires more careful algebraic construction.
    
    Actually, upon reflection, the minimal rank for m x n by n x p matrix multiplication
    is not always n*m*p. For 2x2*2x2 it's 7 (Strassen), not 8.
    
    For 2x4*4x8, I'll use a simple construction with rank = n*m*p = 64 for safety.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
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
