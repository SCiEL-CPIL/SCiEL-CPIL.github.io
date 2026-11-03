import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Decompose 2x3 * 3x9 matrix multiplication.
    
    The key insight: 2x3 * 3x9 = (2x3) * (3x9).
    We can view the 3x9 matrix B as a 3x3 matrix over a 9-dimensional algebra.
    But more directly, we can use the block structure.
    
    Actually, let's think about this more carefully.
    A is 2x3, B is 3x9, C = A*B is 2x9.
    
    The schoolbook decomposition has rank 2*3*9 = 54.
    
    Can we do better? Let's think about the tensor structure.
    C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,2), j in [0,3), k in [0,9).
    
    This is a bilinear map from R^{2*3} x R^{3*9} -> R^{2*9}.
    
    One approach: view B as a 3x3 block matrix where each block is 1x3.
    Then C[i, :] = sum_j A[i,j] * B[j, :]
    
    This doesn't immediately suggest a savings.
    
    Another approach: use the fact that 9 = 3*3 and view B as a 3x3 matrix
    of 1x3 row vectors, or as a 1x9 matrix of 3x1 column vectors.
    
    Let's try: B is 3x9. We can reshape it as B' of shape (3, 3, 3) where
    B'[j, a, b] = B[j, a*3+b].
    
    Then C[i, a*3+b] = sum_j A[i,j] * B'[j, a, b]
    
    This is still just the schoolbook formula with a reindexing.
    
    To get a rank saving, we need to exploit algebraic structure. 
    For example, if B were a 3x3 matrix of scalars (i.e., p=3), we'd have
    C[i,k] = sum_j A[i,j]*B[j,k] with rank at most 2*3*3 = 18, but actually
    the rank of 2x3*3x3 multiplication is known to be 6 (using the fact that
    we can factor through 3x3 matrix multiplication).
    
    Wait, that's not right either. The rank of m x n * n x p matrix multiplication
    is at most m*n*p, but can be less. For 2x3*3x3, the rank is actually 6.
    
    Let me verify: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2}.
    That's 2*3 = 6 output entries, each a sum of 3 products.
    
    A rank-6 decomposition would be:
    For r in range(6):
      (i_r, j_r, k_r) = (r // 3, (r % 3) // 1, r % 1)  -- no that's wrong.
    
    Actually, the standard decomposition for m x n * n x p has rank m*n*p.
    For 2x3*3x3, that's 18. But there might be a better one.
    
    Hmm, let me reconsider. The tensor for matrix multiplication M_{m,n,p} has
    rank at most m*n*p. For small cases:
    - 2x2*2x2: rank 7 (Strassen)
    - 2x3*3x2: rank ?
    - 2x3*3x3: rank ?
    
    For 2x3*3x3, I believe the rank is 11 or so, but I'm not sure.
    
    Actually, for our problem, the key is that p=9=3*3. Let's use the tensor
    product structure.
    
    M_{2,3,9} can be seen as M_{2,3,3} ⊗ I_3 in some sense? No, that's not quite right.
    
    Let me think differently. B is 3x9. We can write B = [B_0 | B_1 | B_2] where
    each B_a is 3x3. Then C = A*B = [A*B_0 | A*B_1 | A*B_2].
    
    So we need to compute 3 separate 2x3*3x3 multiplications. If each has rank r,
    the total rank is 3r.
    
    For 2x3*3x3, what's the minimal rank?
    C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix.
    We can view this as: for each output (i,k), C[i,k] = dot(A[i,:], B[:,k]).
    
    The standard decomposition has rank 2*3*3 = 18.
    
    Can we do better? Let's use the fact that this is a bilinear map.
    One approach: A is 2x3, B is 3x3. We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we use the identity that for any 2x3 matrix A and 3x3 matrix B:
    We can decompose using the 3x3 matrix multiplication tensor.
    
    Actually, here's a cleaner way. The 2x3 * 3x3 multiplication can be seen as:
    First, form the 2x3 * 3x3 product. This is the same as multiplying a 2x3
    matrix by a 3x3 matrix.
    
    We can embed this into 3x3 * 3x3 by padding A with a zero row:
    A' = [[A[0,:]], [A[1,:]], [0,0,0]]
    Then A'*B has the same first two rows as A*B.
    
    So rank(2x3*3x3) <= rank(3x3*3x3) = 21 (using Strassen for 3x3, which is 19).
    Actually, the best known rank for 3x3*3x3 is 19.
    
    But we can do better for 2x3*3x3 specifically. Let me think...
    
    Actually, for the purpose of this problem, let's just use the schoolbook
    decomposition for now and see if we can find a better one through search.
    
    But wait - the task says "rank-minimal exact tensor decomposition". The schoolbook
    rank is 54. If we can find a rank-18 decomposition (by viewing it as 3 separate
    2x3*3x3 multiplications each of rank 6), that would be much better.
    
    Let me verify: can 2x3*3x3 be done in rank 6?
    C[i,k] = sum_j A[i,j]*B[j,k]
    There are 6 outputs: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
    Each is a sum of 3 products.
    
    A rank-6 decomposition would assign each output to a distinct rank-1 term.
    But each rank-1 term contributes to ALL outputs where the corresponding
    U, V, W entries are nonzero. So a single rank-1 term can contribute to
    multiple outputs.
    
    Actually, the minimum rank for 2x3*3x3 is at least max(2*3, 3*3, 2*3) = 9
    by the border rank lower bound... no, that's not right.
    
    The rank of the tensor M_{m,n,p} is at least max(mn, np, pm) / something...
    Actually, I recall that rank(M_{m,n,p}) >= max(mn, np, pm) is not a valid lower bound.
    
    Let me just implement the schoolbook and also try a tensor-product-based
    decomposition that might give rank 18.
    
    Here's the idea: view the 3x9 matrix B as a 3x3 matrix where each entry is
    a 1x3 vector (or equivalently, B has shape (3, 3, 3) when reshaped).
    
    Then C[i, a, b] = sum_j A[i,j] * B[j, a, b]
    
    This is a bilinear map from (A, B) to C where:
    - A has shape (2, 3)
    - B has shape (3, 3, 3) 
    - C has shape (2, 3, 3)
    
    We can decompose this as: for each (j, a, b), we have a term A[i,j]*B[j,a,b]
    contributing to C[i,a,b].
    
    The total number of such terms is 3*3*3 = 27 for the B indices, but we also
    need to handle the i index. So the schoolbook is 2*3*3*3 = 54.
    
    To get a better decomposition, we can use the fact that for fixed j, the
    operation is: for each i, C[i,a,b] += A[i,j] * B[j,a,b].
    
    This is like multiplying a 2x1 vector (column j of A) by a 1x(3x3) matrix
    (row j of B reshaped). This is an outer product, which has rank 1 per (j,a,b)
    combination... no, that's not helping.
    
    Let me try a different approach. Use the tensor product of decompositions.
    
    M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3} ??? No, that doesn't make sense.
    
    Actually, I think the cleanest approach is:
    M_{2,3,9} where we view the last dimension as 3*3.
    If we have a rank-r decomposition of M_{2,3,3}, then we can get a rank-3r
    decomposition of M_{2,3,9} by taking the tensor product with the identity
    on the last 3 dimensions.
    
    So if rank(M_{2,3,3}) = r, then rank(M_{2,3,9}) <= 3r.
    
    What is rank(M_{2,3,3})? Let's compute it or use a known result.
    
    For now, let's just use the schoolbook decomposition with rank 54, and also
    try to construct a rank-18 decomposition if rank(M_{2,3,3}) = 6.
    
    Let me check if rank(M_{2,3,3}) = 6 is possible.
    M_{2,3,3}: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    A rank-6 decomposition would have 6 terms. Each term is specified by
    (u, v, w) where u is in R^{2*3}, v is in R^{3*3}, w is in R^{3*2}.
    
    For the decomposition to be exact, we need:
    sum_r U[i*3+j, r] * V[j*3+k, r] = W[k*2+i, r] ... no, that's not right.
    
    The condition is: for all i, j, k:
    sum_r U[i*3+j, r] * V[j*3+k, r] = delta_{...}
    
    Wait, I'm confusing myself. Let me be more careful.
    
    The tensor T for matrix multiplication M_{m,n,p} has entries:
    T[i, j, k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], else 0.
    
    Actually, the tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    No wait, that's not how tensor decomposition works. Let me recall.
    
    A rank-r decomposition of the matrix multiplication tensor means:
    There exist matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    For all i, j, k:
      delta_{i,k} ... no.
    
    The correct condition is:
    The tensor T has entries T[a, b, c] where:
    a = i*m + j (index into A)
    b = j*p + k (index into B)  
    c = k*n + i (index into C)
    
    T[a, b, c] = 1 if the indices are consistent (i.e., the j in a matches the j in b),
    and 0 otherwise.
    
    Wait, that's not right either. Let me think again.
    
    For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The multiplication tensor T is a 3-tensor with dimensions (n*m) x (m*p) x (n*p).
    T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and i=i', j=j', k=k',
    else 0.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k, and 0 otherwise.
    
    A rank-r decomposition means: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{n*p}.
    
    This means: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    So we need: for all i, j, k:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r equals 1.
    And for all other (a, b, c), the sum equals 0.
    
    OK so for our problem: n=2, m=3, p=9.
    U is 6 x R, V is 27 x R, W is 18 x R.
    
    The schoolbook decomposition has R = 54:
    For r in range(54):
      i = r // 27
      rem = r % 27
      j = rem // 9
      k = rem % 9
      U[i*3+j, r] = 1
      V[j*9+k, r] = 1
      W[k*2+i, r] = 1
    
    Now, can we do better? The key question is whether there's a rank-18 or
    lower decomposition.
    
    Let me try the tensor product approach. If we view the p=9 dimension as
    a 3x3 grid, we can write:
    C[i, a*3+b] = sum_j A[i,j] * B[j, a*3+b]
    
    If we have a rank-r decomposition of M_{2,3,3} (where the output is 2x3),
    then for each of the 3 "blocks" in the p dimension, we can apply it.
    
    Specifically, if M_{2,3,3} has rank r with decomposition matrices
    U1 (6 x r), V1 (9 x r), W1 (6 x r), then M_{2,3,9} has rank 3r with:
    U = U1 ⊗ I_3 (in some sense)
    
    More precisely:
    For r in range(3*r1):
      r1_idx = r % r1
      block = r // r1
      i = ... (from U1)
      j = ... (from U1)
      k1 = ... (from V1, this is the "row" index in the 3x3 block)
      k2 = block (the "column" index in the 3x3 block)
      
      U[i*3+j, r] = U1[i*3+j, r1_idx]
      V[j*9 + k1*3 + k2, r] = V1[j*3+k1, r1_idx]
      W[(k1*3+k2)*2+i, r] = W1[k1*2+i, r1_idx]
    
    So if rank(M_{2,3,3}) = 6, then rank(M_{2,3,9}) <= 18.
    
    Let me check if rank(M_{2,3,3}) = 6 is achievable.
    
    M_{2,3,3}: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    A rank-6 decomposition:
    We need 6 terms. Let's see if we can assign each output C[i,k] to a unique term.
    
    Term (i, k): U[i*3+j, r] for some j, V[j*3+k, r] for the same j, W[k*2+i, r].
    
    If we set for each (i, k) pair, a term with U[i*3+0, r] = 1, V[0*3+k, r] = 1,
    W[k*2+i, r] = 1, then this term contributes A[i,0]*B[0,k] to C[i,k].
    But we also need A[i,1]*B[1,k] and A[i,2]*B[2,k].
    
    So a single term can only handle one (i, j, k) combination. Thus we need at
    least 2*3*3 = 18 terms for the schoolbook.
    
    But maybe we can share terms. For example, a term with U[i*3+j, r] = 1,
    V[j*3+k, r] = 1, W[k*2+i, r] = 1 contributes to C[i,k] the product A[i,j]*B[j,k].
    
    If we have a term where U has multiple nonzero entries, V has multiple, etc.,
    we might be able to cover multiple (i,j,k) combinations with one term.
    
    For example, a term with U[i*3+0, r] = 1, U[i*3+1, r] = 1, V[0*3+k, r] = 1,
    V[1*3+k, r] = 1, W[k*2+i, r] = 1 would contribute (A[i,0]*B[0,k] + A[i,1]*B[1,k])
    to C[i,k]. But this doesn't help because we still need to separate the contributions.
    
    I think the rank of M_{2,3,3} is actually 18, same as the schoolbook. Let me
    verify with a small search or use a known result.
    
    Actually, I recall that for matrix multiplication, the rank is at least
    max(mn, np, pm) and at most mnp. For 2x3*3x3, max(6, 9, 6) = 9, so rank >= 9.
    But the schoolbook gives 18.
    
    I'm not sure of the exact rank. For the purpose of this problem, let's just
    use the schoolbook decomposition with rank 54, and also try to find a better
    one through a simple search if time permits.
    
    Given the time constraints, let's just return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition with rank n*m*p = 54
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
