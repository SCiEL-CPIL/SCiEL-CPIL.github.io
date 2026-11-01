"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # We want to multiply A (2x2) by B (2x14) to get C (2x14).
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # The tensor for this multiplication has indices:
    #   U dimension: n*m = 2*2 = 4 (index = i*m + j = 2*i + j)
    #   V dimension: m*p = 2*14 = 28 (index = j*p + k = 14*j + k)
    #   W dimension: p*n = 14*2 = 28 (index = k*n + i = 2*k + i)
    #
    # Rank: For a 2x2 times 2x14 matrix multiplication, we can decompose this
    # as two independent 1x2 times 2x14 vector-matrix products (one for each
    # row of the output). Each 1x2 * 2x14 product has rank at most 2*14 = 28
    # by schoolbook, but actually we can do better.
    #
    # Actually, let's think more carefully. The multiplication tensor T has
    # entries T[i,j,k] where i in {0,1} (output row), j in {0,1} (inner dim),
    # k in {0,...,13} (output col). The tensor is:
    #   T[i, j, k] = delta_{i,j} * (coefficient for B[j,k] contributing to C[i,k])
    #
    # Wait, the standard matrix multiplication tensor for (n x m) * (m x p) is:
    #   T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that
    # each product term appears exactly once).
    #
    # The rank of this tensor is the minimum number of rank-1 tensors needed.
    # A rank-1 tensor is T[i,j,k] = U[i,j] * V[j,k] * W[k,i] (with appropriate
    # index mappings).
    #
    # For our case: n=2, m=2, p=14.
    # The tensor has 2*2*14 = 56 nonzero entries (all 1s).
    #
    # Schoolbook gives rank 56.
    #
    # Can we do better? Let's think about the structure.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,...,13}.
    #
    # This is a bilinear map. We can view it as:
    # For fixed i, the map (A[i,0], A[i,1]) -> C[i,:] is a 2-dimensional
    # subspace of the 14-dimensional output, parameterized by B[0,:] and B[1,:].
    #
    # Actually, let's think of it differently. The multiplication can be seen as:
    # C = A * B where A is 2x2 and B is 2x14.
    #
    # We can decompose A into its two rows: A = [a_0; a_1] where a_0, a_1 are
    # 1x2 vectors. Then C = [a_0 * B; a_1 * B] where a_i * B is a 1x14 vector.
    #
    # Each a_i * B = a_i[0] * B[0,:] + a_i[1] * B[1,:].
    #
    # So for each output row i, we need to compute a linear combination of the
    # two rows of B with coefficients from row i of A.
    #
    # The key insight: we can share computations between the two rows of A.
    #
    # Let's use the following decomposition. We introduce intermediate variables:
    # s_j = A[0,j] + A[1,j] for j in {0,1}
    # d_j = A[0,j] - A[1,j] for j in {0,1}
    #
    # Then:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This doesn't immediately help. Let me think about tensor rank more carefully.
    #
    # The tensor T is in C^{2x2} ⊗ C^{2x14} ⊗ C^{14x2}.
    # T_{(i,j),(j',k),(k',i')} = 1 if j=j' and i=i' and k=k', else 0.
    #
    # Actually, the standard indexing:
    # U has shape (n*m, rank) = (4, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank) = (28, rank), indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank) = (28, rank), indexed by (k,i) -> k*n+i
    #
    # The decomposition says: for each (i,j,k),
    #   1 = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is a standard matrix multiplication tensor decomposition.
    #
    # For n=2, m=2, p=14, the schoolbook rank is 2*2*14 = 56.
    #
    # Can we achieve rank less than 56? The tensor rank of matrix multiplication
    # is a hard problem. For small cases, we know:
    # - 2x2 * 2x2: rank 7 (Strassen) vs schoolbook 8
    # - 1x2 * 2x14: This is a vector-matrix product. The tensor is in
    #   C^{1x2} ⊗ C^{2x14} ⊗ C^{14x1}. The schoolbook rank is 1*2*14 = 28.
    #   Can we do better? For 1xm * mxp, the rank is at most m*p, and it's
    #   known that the rank is exactly m*p for the standard tensor (since it's
    #   a "slice" of matrix multiplication). Actually, I believe the rank of
    #   the 1xm * mxp tensor is m*p.
    #
    # Wait, let me reconsider. The 1x2 * 2x14 tensor:
    # T_{j,k} (for the single output entry) = 1 for all j in {0,1}, k in {0,...,13}.
    # This is just a 2x14 matrix of 1s. The "rank" in the tensor sense...
    #
    # Actually, for 1xm * mxp, the multiplication is: c_k = sum_j a_j * b_{j,k}.
    # The tensor has entries T[j, (j',k), k'] = delta_{j,j'} * delta_{k,k'}.
    # Hmm, I'm getting confused with the indexing.
    #
    # Let me just use the known result: the rank of the matrix multiplication
    # tensor for (n x m) * (m x p) is at most n*m*p (schoolbook). For our case,
    # that's 56.
    #
    # A key observation: we can factor the problem. Since n=2 is small, we can
    # write the multiplication as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # For each j, the term A[i,j] * B[j,k] can be computed. But we can share
    # the B[j,k] values across different i.
    #
    # Actually, here's a cleaner approach. Note that:
    # C[0,k] = A[0,0] B[0,k] + A[0,1] B[1,k]
    # C[1,k] = A[1,0] B[0,k] + A[1,1] B[1,k]
    #
    # We can introduce for each k:
    # x_k = B[0,k], y_k = B[1,k]
    #
    # Then C[0,k] = A[0,0] x_k + A[0,1] y_k
    # C[1,k] = A[1,0] x_k + A[1,1] y_k
    #
    # This is just a 2x2 matrix multiplication applied to the vector (x_k, y_k)
    # for each k. Since the 2x2 multiplication has rank 7 (Strassen), and we
    # have 14 such independent k values, the total rank would be 14 * 7 = 98,
    # which is worse than schoolbook.
    #
    # So sharing across k doesn't help. The schoolbook rank of 56 is likely
    # optimal or close to it for this problem.
    #
    # Let me just return the schoolbook decomposition with rank 56.
    n, m, p = DIMENSIONS
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
