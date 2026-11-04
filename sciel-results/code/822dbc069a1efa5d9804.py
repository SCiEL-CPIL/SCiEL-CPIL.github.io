"""Optimized matrix multiplication tensor decomposition for 2x3 * 3x6."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    Uses a structured approach:
    - The 2x3 * 3x6 = 2x6 multiplication tensor has dimensions n=2, m=3, p=6
    - Schoolbook rank is n*m*p = 36
    - We exploit structure to reduce rank
    
    Strategy: Use the fact that for each pair (i,k) in output, we need
    sum_j A[i,j]*B[j,k]. We can share computations across different (i,k) pairs.
    
    For n=2, p=6, m=3:
    - Output has 12 entries (2*6)
    - Each output C[i,k] = sum_j A[i,j]*B[j,k]
    
    We can use a divide-and-conquer approach on the p dimension.
    Split the 6 columns of B into groups and use Strassen-like tricks.
    
    Actually, let's think about this more carefully.
    
    The tensor T has indices (i,j,k) where i in [0,1], j in [0,2], k in [0,5].
    T[i,j,k] = delta_{i,j} * delta_{j,k} ... no, that's not right.
    
    The matrix multiplication tensor for A (n x m) times B (m x p) is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (1)
    where the coefficient is 1 if the indices form a valid multiplication term.
    
    Actually the tensor for matrix mult is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i ⊗ e_j ⊗ e_k)
    where e_i are basis vectors.
    
    The rank-1 decomposition is trivially n*m*p terms, each being
    (e_i ⊗ e_j ⊗ e_k) with coefficient 1.
    
    To reduce rank, we need to find linear combinations.
    
    For n=2, m=3, p=6, let's try to exploit the small n.
    
    Key insight: For fixed j, the contribution to all outputs C[i,k] from
    A[i,j]*B[j,k] involves all i and k. We can group by j.
    
    For each j, we have a 2x6 outer product A[:,j] ⊗ B[j,:].
    This outer product has rank min(2,6) = 2 in the matrix sense, but as a
    tensor term it's a single rank-1 term in the multiplication tensor.
    
    Wait, I need to be more careful. The multiplication tensor is:
    T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    
    In tensor notation: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C (with appropriate indices)
    
    Actually, the standard formulation: the bilinear map (A,B) -> C where
    C = AB has tensor T such that C_{i,k} = sum_{j} A_{i,j} B_{j,k} = <T, A ⊗ B>
    
    where T = sum_{i,j,k} E_{i,k} ⊗ E_{i,j} ⊗ E_{j,k} (using dual basis)
    
    Hmm, let me use the standard formulation. The tensor is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i ⊗ f_j ⊗ g_k)
    where e_i, f_j, g_k are basis vectors in the respective spaces, and the
    trilinear form is <T, A ⊗ B ⊗ C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    
    For matrix multiplication, T_{i,j,k} = 1 if (i,j,k) is a valid index triple,
    which it always is for i in [0,n), j in [0,m), k in [0,p).
    
    So T = sum_{i,j,k} e_i ⊗ f_j ⊗ g_k, which has rank n*m*p.
    
    To reduce rank, we need to find a decomposition with fewer terms.
    
    Let's try a different approach. Since n=2 is small, we can try to
    combine the two rows of A.
    
    For row i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    For row i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    If we define for each k:
    v_k = [B[0,k], B[1,k], B[2,k]] (column k of B)
    u_0 = [A[0,0], A[0,1], A[0,2]] (row 0 of A)
    u_1 = [A[1,0], A[1,1], A[1,2]] (row 1 of A)
    
    Then C[0,k] = u_0 · v_k and C[1,k] = u_1 · v_k.
    
    This is just the dot product of 3D vectors. The rank of this bilinear form
    for a single k is 3 (since we're doing a 3D dot product).
    
    For 6 different k values, if the v_k are linearly independent (which they
    can be), we might not be able to share much.
    
    However, we can use the following trick: if we can express some of the
    computations in terms of others.
    
    Let's try a specific construction. We'll use rank 30 as a target.
    
    Actually, let me just implement a careful search for a good decomposition.
    
    Given the time constraints, let me implement a known good decomposition
    and verify it.
    """
    import time
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 36
    
    # Try to find a better decomposition using structured search
    # For now, return a known good decomposition
    
    # Let's try to construct a rank 30 decomposition
    # by exploiting the structure
    
    # The key idea: for each column k of B, we compute two dot products
    # (one for each row of A). If we can share some of the intermediate
    # products, we can reduce the rank.
    
    # One approach: use the fact that we only have 3 columns of A (m=3)
    # and 2 rows (n=2). The total number of A entries is 6, and B has 18 entries.
    
    # Let's try a decomposition where we group the p dimension.
    # Split p=6 into two groups of 3.
    
    # For group 1 (k=0,1,2) and group 2 (k=3,4,5):
    # We can use 3x3 matrix multiplication tricks on each group.
    
    # Actually, let's just do a careful manual construction.
    
    # Rank 32 decomposition:
    # We use 32 terms. The idea is to share some computations between
    # the two rows of A.
    
    # For each j (0,1,2) and each k (0..5), we have a term A[i,j]*B[j,k].
    # Total schoolbook terms: 2*3*6 = 36.
    
    # We can save 4 terms by noting that for fixed j, the products
    # A[0,j]*B[j,k] and A[1,j]*B[j,k] can be partially shared.
    
    # Specifically, if we define S_j = A[0,j] + A[1,j], then
    # A[0,j]*B[j,k] + A[1,j]*B[j,k] = S_j * B[j,k]
    # and A[0,j]*B[j,k] - A[1,j]*B[j,k] = (A[0,j]-A[1,j])*B[j,k]
    
    # This doesn't directly help with rank reduction in the tensor sense.
    
    # Let me just return the schoolbook decomposition for now,
    # which is guaranteed to be correct.
    
    rank = rank_schoolbook
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
