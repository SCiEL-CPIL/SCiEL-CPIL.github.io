"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U @ diag(1) @ V^T == tensor contracted with W gives correct multiplication."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k are valid indices for the multiplication, else 0.
    # Actually, for matrix multiplication (n x m) @ (m x p) -> (n x p):
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # The standard tensor has T[i,j,k] = 1 for all valid (i,j,k) triples? No.
    # 
    # Let me reconsider. The matrix multiplication tensor T has indices:
    # T[i, j, k] where i is row of A (n), j is column of A / row of B (m), k is column of B (p)
    # T[i, j, k] = 1 for all valid i, j, k.
    # 
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # For schoolbook: each (i,j,k) triple gets its own term, so U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # for a unique c per (i,j,k).
    
    rank = len(U[0]) if U else 0
    # Build the full tensor from decomposition
    # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Use numpy for efficiency
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # This is a 3-mode contraction. We can do it as:
    # For each i,j,k: sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    
    # Efficient way: 
    # Step 1: M[j,k,c] = V_3d[j,k,c] * W_3d[k,j,c]... no wait, W is indexed [k*n+i, c]
    # W_3d[k, i, c] corresponds to W[k*n+i, c]
    
    # Let's just do it directly for small sizes
    T_decomp = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_decomp[i, j, k] = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
    
    # The target tensor is all ones (for valid indices in matrix mult)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_decomp, T_target, atol=1e-9)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Strategy: Use the algebraic structure. A 2x3 * 3x6 multiplication can be viewed as:
    - The 3x6 matrix is a linear combination of two 3x3 matrices
    - Use rank-7 decomposition for 3x3*3x3 (Strassen-like)
    
    Actually, let's think more carefully. The multiplication is (2x3) @ (3x6) -> (2x6).
    
    Key insight: We can split the 3x6 matrix into two 3x3 blocks: B = [B1, B2] where B1, B2 are 3x3.
    Then A @ B = [A @ B1, A @ B2] where A is 2x3.
    
    So we need to compute A @ B1 and A @ B2, each is (2x3) @ (3x3) -> (2x3).
    
    For (2x3) @ (3x3): We can think of this as computing for each of the 2 rows of A,
    the product of a 1x3 vector with a 3x3 matrix, giving 2x3 results.
    
    Alternatively, think of it as: A is 2x3, B is 3x3. The product C = A@B is 2x3.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is like 2 independent (1x3) @ (3x3) multiplications, or equivalently a (2x3)@(3x3) problem.
    
    The tensor for (2x3)@(3x3) has dimension 2*3*3 = 18 entries in the schoolbook.
    
    For the full problem (2x3)@(3x6), the schoolbook rank is 2*3*6 = 36.
    
    Now, can we do better? 
    
    Approach: View the 3x6 as two 3x3 blocks. The multiplication tensor for (2x3)@(3x6) 
    is the "concatenation" (in the second matrix's column dimension) of two (2x3)@(3x3) tensors.
    
    If we can find a rank-r decomposition for (2x3)@(3x3), then the rank for (2x3)@(3x6) 
    would be at most 2r (using the same decomposition for both blocks, just with different
    columns in the output).
    
    Actually, more precisely: if T_{2,3,3} has rank r, then T_{2,3,6} can be decomposed with rank 2r
    by using the same rank-r decomposition for each 3x3 block.
    
    So the question reduces to: what is the rank of the (2x3)@(3x3) multiplication tensor?
    
    The (2x3)@(3x3) tensor: indices (i, j, k) with i in [0,2), j in [0,3), k in [0,3).
    T[i,j,k] = 1 for all valid (i,j,k).
    
    This is a 2x3x3 tensor. The schoolbook rank is 18.
    
    Can we do better? For a 2x3x3 tensor, the rank is at most 18. 
    Known results: The rank of the 2x3x3 matrix multiplication tensor...
    
    Actually, for m x n x p matrix multiplication, the tensor rank is at most m*n*p.
    For 2x3x3, that's 18. Can we do better?
    
    One approach: Group the 2 rows. Think of A as a 2x3 matrix. 
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can write this as: for each k, C[:,k] = A @ B[:,k] (a 2x3 times 3x1 = 2x1).
    Or: for each i, C[i,:] = A[i,:] @ B (a 1x3 times 3x3 = 1x3).
    
    Let's try a different decomposition. Consider:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Define intermediate variables:
    M1 = A[0,0]*B[:,0] + A[0,1]*B[:,1] + A[0,2]*B[:,2]  (this is row 0 of C)
    M2 = A[1,0]*B[:,0] + A[1,1]*B[:,1] + A[1,2]*B[:,2]  (this is row 1 of C)
    
    That's just the standard computation.
    
    Let me try to use a rank-7 approach for 3x3x3 (Strassen) and adapt it.
    
    Actually, for the specific case of 2x3 x 3x6, let me try a direct approach:
    
    The output is 2x6. Each output element C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k].
    
    Total output elements: 12. Each requires a sum of 3 products.
    
    Schoolbook: 12 * 3 = 36 multiplications (rank 36).
    
    Can we share computations? 
    
    Consider that B is 3x6. Let's denote B[j,k] for j in 0..2, k in 0..5.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    ...
    
    Notice that C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] for k=0..5.
    This is A[0,:] @ B[:,k] for each k.
    
    Similarly C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k].
    
    So we're computing 2 rows, each being a linear combination of 3 rows of B with coefficients from A.
    
    This is equivalent to: C = A @ B where A is 2x3 and B is 3x6.
    
    One approach: Use the fact that we can compute A @ B by first computing some intermediate
    products and then combining.
    
    Let me try a rank-18 decomposition (half of schoolbook) by exploiting structure.
    
    Actually, let me think about this differently. The tensor T[i,j,k] for (2x3)@(3x6) 
    is 1 for all (i,j,k) in [0,2)x[0,3)x[0,6).
    
    This tensor can be seen as the "Khatri-Rao" or some other product structure.
    
    T_{2,3,6} = T_{2,3,3} \otimes T_{1,1,2} in some sense? No, that's not quite right.
    
    Actually, T_{2,3,6}[i,j,k] = T_{2,3,3}[i,j,k mod 3] * T_{1,1,2}[0,0, k // 3]
    where T_{1,1,2} is just the identity (rank 2).
    
    So T_{2,3,6} is the "tensor product" (in the appropriate sense) of T_{2,3,3} and a rank-2 tensor.
    This means rank(T_{2,3,6}) <= rank(T_{2,3,3}) * 2.
    
    So if we find a rank-r decomposition for T_{2,3,3}, we get rank 2r for T_{2,3,6}.
    
    Now, what's the rank of T_{2,3,3}?
    
    T_{2,3,3}[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    This is a 2x3x3 tensor. Let's try to find a low-rank decomposition.
    
    One idea: For each k, the 2x3 slice T[:,:,k] is all ones. So all 3 slices are the same.
    This means T[i,j,k] = S[i,j] * 1 where S is the all-ones 2x3 matrix.
    So T = S \otimes 1 (outer product with the all-ones vector of length 3).
    
    The rank of S (2x3 all-ones) is 1. So T has rank at most 3 (using the 3 basis vectors for the k dimension).
    
    Wait, that's not right for tensor rank. Let me reconsider.
    
    T[i,j,k] = 1 for all i,j,k. This is the all-ones tensor of size 2x3x3.
    
    The tensor rank of the all-ones tensor of size a x b x c is at most min(a,b,c) in general? No.
    For a 2x3x3 all-ones tensor, the rank is actually 3 (or less).
    
    Let's verify: T[i,j,k] = sum_{c=0}^{2} u_c[i] * v_c[j] * w_c[k]
    
    If we set u_0 = u_1 = u_2 = [1,1]^T, v_0 = v_1 = v_2 = [1,1,1]^T, and w_c is the c-th standard basis vector in R^3, then:
    T[i,j,k] = sum_c 1 * 1 * delta_{c,k} = 1.
    
    Yes! So the all-ones 2x3x3 tensor has rank 3.
    
    Therefore, T_{2,3,6} has rank at most 3 * 2 = 6!
    
    Let me construct this explicitly.
    
    For the (2x3)@(3x6) multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    Decomposition with rank 6:
    For c in 0..2 (corresponding to the j index... wait, let me redo this).
    
    Actually, I need to be more careful. The tensor for matrix multiplication (n x m) @ (m x p) -> (n x p) 
    is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Note the index mappings:
    - U has indices (i*m + j, c) where i in [0,n), j in [0,m)
    - V has indices (j*p + k, c) where j in [0,m), k in [0,p)
    - W has indices (k*n + i, c) where k in [0,p), i in [0,n)
    
    For n=2, m=3, p=6:
    - U: 6 x rank
    - V: 18 x rank
    - W: 12 x rank
    
    Now, T[i,j,k] = 1 for all (i,j,k).
    
    Let's try rank 6. For each (j, block) where j in 0..2 and block in 0..1 (for the two 3-column blocks of the 6 columns):
    
    Actually, let me use the factorization: k = 3*block + k' where block in {0,1}, k' in {0,1,2}.
    
    T[i,j,k] = T[i,j,3*block+k'] = 1.
    
    We can write: T[i,j,k] = sum_{j'} delta_{j,j'} * sum_{block} delta_{block, block}
    
    Hmm, let me just directly construct a rank-6 decomposition.
    
    For c = 0,1,2 (corresponding to j=0,1,2) and for each of the 2 blocks:
    
    Let's use 6 terms: for j in {0,1,2} and block in {0,1}:
    
    Term (j, block):
    - U[i*m + j, c] = 1 for all i (since we need T[i,j,k]=1 for all i)
      So U[i*3 + j, c(j,block)] = 1 for i=0,1
    - V[j*p + k, c] = 1 for k in the block (k = 3*block, 3*block+1, 3*block+2), 0 otherwise
      So V[j*6 + 3*block + k', c(j,block)] = 1 for k'=0,1,2
    - W[k*n + i, c] = 1 for all i and k in the block
      So W[(3*block+k')*2 + i, c(j,block)] = 1 for i=0,1, k'=0,1,2
    
    Let's verify: For a given (i,j,k) where k = 3*block + k':
    T_decomp[i,j,k] = sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c]
    
    Only the term c = (j, block) contributes (assuming unique c for each (j,block)):
    = U[i*3+j, c(j,block)] * V[j*6+3*block+k', c(j,block)] * W[(3*block+k')*2+i, c(j,block)]
    = 1 * 1 * 1 = 1.
    
    For other c values, at least one factor is 0.
    
    So this gives a valid rank-6 decomposition!
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 6
    U = [[0] * rank for _ in range(n * m)]  # 6 x 6
    V = [[0] * rank for _ in range(m * p)]  # 18 x 6
    W = [[0] * rank for _ in range(p * n)]  # 12 x 6
    
    # For each j in 0..2 and block in 0..1:
    for j in range(m):
        for block in range(p // 3):  # 2 blocks
            c = j * (p // 3) + block
            # U: for all i, U[i*m + j, c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
            # V: for k in block, V[j*p + k, c] = 1
            for k_prime in range(3):
                k = block * 3 + k_prime
                V[j * p + k][c] = 1
            # W: for all i and k in block, W[k*n + i, c] = 1
            for k_prime in range(3):
                k = block * 3 + k_prime
                for i in range(n):
                    W[k * n + i][c] = 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
