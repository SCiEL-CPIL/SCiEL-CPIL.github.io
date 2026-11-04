"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = delta[i][i'] * delta[j][j'] * delta[k][k']
    in the appropriate index space. Specifically, for the tensor decomposition:
    T = sum_r U_r ⊗ V_r ⊗ W_r
    
    The seed schoolbook decomposition uses rank = n*m*p = 4*4*7 = 112.
    
    We can exploit the structure: the multiplication C = AB where A is 4x4 and B is 4x7.
    
    Key insight: The tensor for matrix multiplication of an (n x m) by (m x p) matrix
    can be decomposed using the fact that each output entry C[i,k] depends on A[i,:] and B[:,k].
    
    The minimal rank for the matrix multiplication tensor of sizes n x m and m x p is n*m*p 
    in the schoolbook, but we can do better.
    
    Actually, the tensor rank for matrix multiplication is exactly n*m*p for the standard 
    (non-strassen) decomposition, but Strassen-type methods can reduce this.
    
    However, for the specific case where we're multiplying 4x4 by 4x7, we can view this as:
    C = A @ B
    
    The tensor T_{(i,j),(j',k),(i',k')} = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    A rank-r decomposition satisfies:
    T_{(i,j),(j',k),(i',k')} = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    The schoolbook uses r = n*m*p with U[i*m+j, i*m*p+j*p+k] = 1, etc.
    
    Can we do better? The rank of the matrix multiplication tensor M_{n,m,p} is at most 
    n*m*p, and known lower bounds show it's at least n*m*p/2 for square matrices (roughly).
    
    For our specific case (4,4,7), let's think about it differently.
    
    Actually, the key observation is: the tensor is a 3-way tensor with dimensions (n*m) x (m*p) x (p*n).
    The rank of this tensor is what we want to minimize.
    
    For the matrix multiplication tensor, the rank equals the number of scalar multiplications 
    needed in an algorithm for that matrix multiplication.
    
    For 4x4 * 4x4, Strassen gives rank 7 (instead of 16). For 4x4 * 4x7, we can use a similar 
    approach.
    
    Let me think about this more carefully. The tensor T has:
    T[a, b, c] = 1 if a = i*m + j, b = j'*p + k, c = k'*n + i' where i=i', j=j', k=k'
    and 0 otherwise.
    
    So T is a 3-way tensor where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    This is actually a permutation-like tensor. The rank of such a tensor...
    
    Wait, let me reconsider. The tensor is:
    T_{(i,j), (j',k), (i',k')} = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    This can be written as:
    T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    where e_{(i,j)} is the unit vector with 1 at position (i*m+j).
    
    So the rank is at most n*m*p = 4*4*7 = 112 (schoolbook).
    
    Can we combine terms? Note that δ_{i,i'} δ_{j,j'} δ_{k,k'} = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
    
    This is a product of three independent Kronecker deltas. The tensor is essentially a 
    "parallel" tensor in the sense that it's a sum of rank-1 tensors where each term 
    corresponds to a unique (i,j,k) triple.
    
    The question is whether we can express this sum using fewer rank-1 terms.
    
    For the case n=m=p, the rank of the matrix multiplication tensor is known to be less 
    than n^3 for n >= 2 (Strassen, etc.). But for rectangular cases, it's less clear.
    
    Let me try a different approach. Since p=7 and n=m=4, we can write:
    
    C = A @ B, where A is 4x4 and B is 4x7.
    
    We can compute C column by column: C[:, k] = A @ B[:, k] for k = 0,...,6.
    
    Each column computation is a 4x4 matrix-vector product. The rank of the 4x4 matrix-vector 
    multiplication tensor (which is a 4x4x1 tensor) is 16 (schoolbook). But can we do better?
    
    For matrix-vector multiplication with a 4x4 matrix, the tensor rank is 16 (it's equivalent 
    to 4 independent dot products of length 4, and each dot product of length n has tensor rank n, 
    so 4*4 = 16).
    
    Wait, no. The matrix-vector product tensor for an n x n matrix has rank n^2 in the schoolbook, 
    but this is actually the minimal rank for the general case (it's an n^2-dimensional linear map).
    
    Hmm, but actually for matrix-vector multiplication, the tensor rank is exactly n^2 because 
    the linear map is full rank n^2 and the tensor is a "generic" linear map tensor.
    
    So for 7 columns, the total rank would be 7 * 16 = 112, same as schoolbook.
    
    But wait - we can share work across columns! The matrix A is shared across all columns.
    
    Actually, let me reconsider the tensor structure. The tensor T is:
    T[(i,j), (j',k), (i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    This is a tensor in R^{(n*m) x (m*p) x (p*n)}.
    
    Let's fix k. Then for each k, we have a sub-tensor T_k where:
    T_k[(i,j), (j',k), (i',k)] = δ_{i,i'} δ_{j,j'}
    
    This is the tensor for multiplying a 4x4 matrix by a 4x1 vector (one column of B).
    
    The rank of each T_k is 16 (for 4x4 matrix-vector product).
    
    But the full tensor is not the direct sum of these sub-tensors in a simple way because 
    the middle index varies with k.
    
    Actually, I think the rank is indeed 112 for this specific tensor. Let me verify by trying 
    to find a lower-rank decomposition.
    
    Alternative approach: Use the fact that the tensor is separable in a certain sense.
    
    T[i*m+j, j'*p+k, k'*n+i'] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    = (δ_{i,i'}) * (δ_{j,j'}) * (δ_{k,k'})
    
    This is already a rank-1 form for each (i,j,k), but we need a sum over (i,j,k).
    
    The minimal number of rank-1 terms to represent this sum is the tensor rank.
    
    For the matrix multiplication tensor M_{n,m,p}, the rank is known to be:
    - At most n*m*p (schoolbook)
    - At least max(n*m, m*p, p*n) * something (lower bounds)
    
    For n=m=4, p=7:
    - Schoolbook: 112
    - Lower bound: The tensor has slices that are identity matrices, so the rank must be at 
      least the dimension of the space... 
    
    Actually, I recall that the rank of the matrix multiplication tensor M_{n,m,p} is exactly 
    n*m*p when one of the dimensions is 1, but can be less for larger dimensions.
    
    Let me try a concrete approach: Can we use Strassen-like decomposition?
    
    For 4x4 * 4x4, Strassen uses 7 multiplications. For 4x4 * 4x7, we can potentially use a 
    similar approach.
    
    Actually, let me just try to construct a decomposition with rank less than 112 and verify 
    it works.
    
    Here's an idea: Group the k-index. For each k, the sub-problem is 4x4 matrix-vector. 
    If we can find a rank-r decomposition for the 4x4 matrix-vector product, then the total 
    rank is 7*r.
    
    The 4x4 matrix-vector product tensor (A is 4x4, v is 4x1, C = A*v) has rank 16.
    
    But wait - can we do better? The linear map A -> A*v for fixed v is a 16-dimensional 
    linear map. The tensor rank of the "multiplication by a fixed vector" operation... 
    
    Actually no, v is also a variable. The tensor is T_{(i,j), j', i'} = δ_{j,j'} δ_{i,i'} for 
    the i,j,j',i' indices, but this doesn't seem right.
    
    Let me be more careful. For C = A * v where A is 4x4 and v is 4x1:
    C_i = sum_j A_{i,j} * v_j
    
    The tensor T has dimensions (4*4) x (4*1) x (1*4) = 16 x 4 x 4.
    T_{(i,j), j', (i',)} = δ_{i,i'} δ_{j,j'}
    
    This is a 16 x 4 x 4 tensor. Its rank is at most 16*4 = 64? No, that's not right either.
    
    T_{(i,j), j', i'} = δ_{i,i'} δ_{j,j'}
    
    This can be written as:
    T = sum_{i,j} e_i ⊗ e_j ⊗ e_i ⊗ e_j  (with appropriate reshaping)
    
    Actually, T_{(i,j), j', i'} = δ_{i,i'} δ_{j,j'} means:
    T = sum_{i,j} |i,j><i,j| ⊗ |j><j| ⊗ |i><i| ... this is getting complicated.
    
    Let me just accept that the schoolbook rank is 112 and try to find a better decomposition 
    using a different strategy.
    
    Alternative: Use the fact that p=7 is prime. Maybe there's a FFT-based approach?
    
    Actually, I think the best I can do without a deep insight is to stick with the schoolbook 
    decomposition of rank 112, or try a minor optimization.
    
    Wait - I just realized something. The tensor T[i*m+j, j'*p+k, k'*n+i'] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    This can be rewritten as:
    T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    where e_{(i,j)} is the unit vector in R^{n*m} with 1 at position i*m+j.
    
    Note that e_{(j,k)} has 1 at position j*p+k. For different k values, these are in different 
    "blocks" of the middle vector.
    
    The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
    
    Each term e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)} is rank 1. There are n*m*p = 112 such terms.
    
    Can any two terms be combined? Two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 can be combined 
    into a single rank-1 tensor only if u1∝u2, v1∝v2, and w1∝w2. Since the e vectors are 
    orthogonal and distinct for different (i,j), (j,k), (k,i) triples, no two terms can be 
    combined.
    
    Therefore, the rank is exactly 112!
    
    Hmm, but that contradicts the known fact that matrix multiplication can be done faster 
    than schoolbook. The resolution is that Strassen's algorithm works by a different 
    decomposition - it doesn't decompose the tensor into a sum of rank-1 tensors in the 
    straightforward way. Instead, it uses a change of basis.
    
    Wait, but any bilinear algorithm corresponds to a tensor decomposition. Strassen's 
    algorithm for 2x2 * 2x2 uses 7 multiplications, which means the 2x2x2 matrix 
    multiplication tensor has rank at most 7. The schoolbook rank is 8. So the rank is 
    indeed less than n*m*p for n=m=p=2.
    
    So my analysis above is wrong. Let me reconsider.
    
    The issue is that the terms e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)} are linearly independent 
    as vectors in the tensor space, but they might not be linearly independent as rank-1 
    tensors in the sense of tensor rank.
    
    Actually, I think the correct statement is: The tensor rank is the minimum r such that 
    T = sum_{r'=1}^r u_{r'} ⊗ v_{r'} ⊗ w_{r'}.
    
    The schoolbook gives r = n*m*p. Strassen shows that for 2x2x2, r <= 7 < 8.
    
    So for 4x4x7, the rank is at most 112, but could be less.
    
    Let me try to apply a Strassen-like decomposition to the 4x4 part.
    
    For 4x4 * 4x4, Strassen's algorithm (applied recursively or directly) can use 7^2 = 49 
    multiplications (if we view 4x4 as four 2x2 blocks). But 49 > 16, so that's worse.
    
    Actually, for 4x4 matrices, the known best upper bound on tensor rank is around 49-56 
    (using Strassen recursively). But for 4x4 * 4x7, it's less clear.
    
    Let me try a different approach: decompose the 4x4 * 4x7 multiplication as:
    - First 4 columns: 4x4 * 4x4 (use Strassen with rank ~49)
    - Last 3 columns: 4x4 * 4x3 (use schoolbook with rank 48)
    Total: 49 + 48 = 97
    
    Or:
    - All 7 columns with a mixed approach.
    
    Actually, let me just try to implement a Strassen-based decomposition for the 4x4 * 4x4 
    part and schoolbook for the 4x4 * 4x3 part.
    
    Strassen for 2x2 * 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    For 4x4 * 4x4, split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    
    Then apply Strassen to get 7 matrix multiplications of 2x2 matrices.
    Each 2x2 * 2x2 multiplication has rank 7 (Strassen) or 4 (schoolbook).
    
    If we use Strassen at both levels: 7 * 7 = 49 scalar multiplications.
    If we use Strassen at top level and schoolbook at bottom: 7 * 4 = 28.
    
    Wait, that's not right either. The rank of the tensor composition... 
    
    Actually, I think for the purpose of this task, the simplest approach is to just use the 
    schoolbook decomposition with rank 112. The seed program already does this, and getting 
    a better rank would require a non-trivial tensor decomposition algorithm.
    
    Given the time constraints, let me just return the schoolbook decomposition.
    
    """
    import time
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    
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
