"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    The strategy is to use the block-diagonal structure:
    - The 3x5 * 5x7 multiplication can be seen as 3 independent row-vector multiplications
    - Each 1x5 * 5x7 tensor has rank 5 (using standard basis)
    - Total rank = 3 * 5 = 15
    
    However, we need to be careful about the tensor structure. Let's think about this more carefully.
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    Actually, the multiplication tensor M_{i,j,k} = 1 for all valid i,j,k (it's the structure tensor).
    
    Wait, let me reconsider. The matrix multiplication tensor is defined such that:
    <M, A ⊗ B> = sum_{i,j,k} M_{i,j,k} * A_{i,j} * B_{j,k} = sum_{i,k} C_{i,k}
    
    where C = A*B. So M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    
    No, that's not right either. The correct definition is:
    M_{i,j,k} is the coefficient such that C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    So the tensor is T[i,k,j] = delta (it's 1 when we multiply A[i,j] and B[j,k] to get C[i,k])
    
    Actually, the standard way: The multiplication tensor mu_{n,m,p} has entries:
    mu_{i,j,k} = 1 if the index corresponds to A[i,j] * B[j,k] contributing to C[i,k]
    
    More precisely, mu is a tensor in (R^{n x m})^* ⊗ (R^{m x p})^* ⊗ (R^{n x p})
    where mu(e_{i,j} ⊗ e_{j',k} ⊗ e_{i',k'}) = 1 if i=i', j=j', k=k'
    
    So mu_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    This tensor has rank at most n*m*p (schoolbook), but can be lower.
    
    For the case n=3, m=5, p=7:
    Schoolbook rank = 3*5*7 = 105
    
    Can we do better? 
    
    One approach: use the fact that we can share computations.
    
    Actually, for rectangular matrix multiplication, the rank is at least max(n*p, m*p, n*m) = max(21, 35, 15) = 35.
    
    And it's known that the rank of matrix multiplication tensor mu_{n,m,p} satisfies:
    max(n*p, m*p, n*m) <= rank <= n*m*p
    
    For n=3, m=5, p=7: lower bound is max(21, 35, 15) = 35.
    
    So we need at least rank 35. Can we achieve rank 35?
    
    One construction: use the "block diagonal" approach where we treat each column of A independently.
    Actually, let me think of a simpler construction.
    
    Consider: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as: for each j, we compute the outer product of column j of A (as a vector of length n) and row j of B (as a vector of length p), then sum them.
    
    So C = sum_j (A[:,j] ⊗ B[j,:])
    
    Each term A[:,j] ⊗ B[j,:] is a rank-1 matrix (outer product of two vectors).
    
    So we have C = sum_{j=1}^{m} (A[:,j] ⊗ B[j,:])
    
    This gives us a decomposition of rank m = 5? No, that's not right because A[:,j] is a vector and B[j,:] is a vector, but they're not rank-1 in the tensor sense.
    
    Let me reconsider the tensor decomposition.
    
    The multiplication tensor mu has a rank-r decomposition if we can write:
    mu = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    
    where u_t is in (R^{n*m})^*, v_t is in (R^{m*p})^*, w_t is in R^{n*p}.
    
    For the schoolbook decomposition, r = n*m*p, and each term corresponds to a single (i,j,k) triple.
    
    To get a lower rank, we need to find a more efficient decomposition.
    
    One known result: for square matrices, the rank is between n^2 and n^3, and for n=2, the rank is 7 (Strassen).
    
    For rectangular matrices, there are various constructions. Let me try to construct one for n=3, m=5, p=7.
    
    Actually, let me try a simple approach: use the fact that we can group terms.
    
    For each pair (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is an inner product of two vectors of length m. The inner product of two m-vectors can be computed with rank m (using the standard basis), but can it be done with lower rank?
    
    No, the inner product tensor (sum_j x_j * y_j) has rank m over a field (it's the identity tensor of size m).
    
    So for each (i,k) pair, we need rank m to compute the inner product. But we can share the computation across different (i,k) pairs.
    
    Specifically, for fixed j, the term A[i,j] * B[j,k] appears in C[i,k] for all i,k. So we can write:
    
    C = sum_j (A[:,j] ⊗ B[j,:])
    
    where A[:,j] is the j-th column of A (length n) and B[j,:] is the j-th row of B (length p).
    
    Now, A[:,j] ⊗ B[j,:] is a matrix of size n x p, which is the outer product of two vectors. As a tensor in (R^n)^* ⊗ (R^p)^* ⊗ (R^{n x p}), this is rank 1.
    
    But wait, we need to express this in terms of the full tensors. Let me be more careful.
    
    Let's index the tensors:
    - U has shape (n*m, rank): U[i*m + j, t]
    - V has shape (m*p, rank): V[j*p + k, t]
    - W has shape (p*n, rank): W[k*n + i, t]
    
    The decomposition is: sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t] should give the multiplication tensor.
    
    Actually, the correct condition is:
    For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if the indices match appropriately, 0 otherwise.
    
    Wait, I need to get the indexing right. The multiplication tensor M[i,j,k] = 1 (it's always 1 for valid indices), and the decomposition should satisfy:
    
    M[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    No, that's not right. Let me look at the schoolbook decomposition:
    
    For each (i,j,k), we have a term with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    
    So the product U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t] should be 1 if i=i', j=j', k=k', and 0 otherwise.
    
    This means the tensor is: sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    
    and this should equal the multiplication tensor, which has entries:
    M[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if i=i', j=j', k=k'
    
    So M is a tensor where M[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for the same i,j,k, and 0 otherwise.
    
    In other words, M[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k, and M[a,b,c] = 0 otherwise.
    
    So the schoolbook decomposition has rank n*m*p, with one term for each (i,j,k).
    
    To get a lower rank, we need to find U, V, W with fewer columns such that:
    sum_t U[(i*m+j), t] * V[(j*p+k), t] * W[(k*n+i), t] = 1 for all i,j,k
    and the sum is 0 for all other index combinations.
    
    This is a constrained problem. Let me try to construct a solution with rank m = 5.
    
    Idea: For each j, define a term that captures all i,k for that j.
    
    Let t = j (so rank = m = 5).
    
    Define:
    U[i*m+j, t] = 1 if t = j, else 0. So U[i*m+j, j] = 1 for all i, and 0 otherwise.
    V[j'*p+k, t] = 1 if t = j', else 0. So V[j*p+k, j] = 1 for all k, and 0 otherwise.
    W[k'*n+i', t] = 1 if t = j, but we need W to depend on i and k.
    
    Wait, this doesn't work because W needs to pick out the specific i,k for each j.
    
    Let me try a different approach. 
    
    Actually, I realize that the minimum rank for matrix multiplication mu_{n,m,p} is at least max(n*p, m*p, n*m) = max(21, 35, 15) = 35.
    
    And there's a construction that achieves rank = m * max(n, p) when one dimension is small.
    
    For n=3, m=5, p=7: rank = 5 * max(3, 7) = 5 * 7 = 35.
    
    Let me try to construct this.
    
    Construction: For each j in {0,...,m-1} and each k in {0,...,p-1}, define a term.
    
    Wait, that's m*p = 35 terms.
    
    Let me define:
    For t = (j, k) where j in [0,m), k in [0,p):
    
    U[i*m+j', t] = delta_{j,j'} * 1  (for all i)
    V[j'*p+k', t] = delta_{j,j'} * delta_{k,k'}
    W[k''*n+i', t] = delta_{k,k''} * 1  (for all i')
    
    Then sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k''*n+i', t]
    = sum_{j,k} delta_{j,j} * 1 * delta_{j,j'} * delta_{k,k} * delta_{k,k''} * 1
    = sum_{j,k} delta_{j,j'} * delta_{k,k''}
    = 1 if j=j' and k=k'', else 0
    
    But we want this to be 1 if i=i', j=j', k=k'', and 0 otherwise. The i and i' indices are not being checked!
    
    So this construction doesn't work. We need to also check i=i'.
    
    Let me try:
    For t = (j, k) where j in [0,m), k in [0,p):
    
    U[i*m+j', t] = delta_{j,j'} * 1
    V[j'*p+k', t] = delta_{j,j'} * delta_{k,k'}
    W[k''*n+i', t] = delta_{k,k''} * delta_{i,i'}  -- but W can't depend on i' in this way because W is indexed by (k''*n+i'), so W[k''*n+i', t] can depend on i'.
    
    Wait, W[k''*n+i', t] is just a value at position (k''*n+i'), so it can depend on both k'' and i'.
    
    Let me redefine:
    For t = (j, k) where j in [0,m), k in [0,p):
    
    U[i*m+j', t] = delta_{j,j'}  (independent of i)
    V[j'*p+k', t] = delta_{j,j'} * delta_{k,k'}
    W[k''*n+i', t] = delta_{k,k''}  (independent of i')
    
    Then the product is:
    U[i*m+j, t] * V[j'*p+k, t] * W[k''*n+i', t]
    = delta_{j,j} * delta_{j,j'} * delta_{k,k} * delta_{k,k''}
    = delta_{j,j'} * delta_{k,k''}
    
    Summing over t=(j,k):
    sum_{j,k} delta_{j,j'} * delta_{k,k''} = 1 if j=j' and k=k'', else 0
    
    But we need it to be 1 if i=i', j=j', k=k'', and 0 otherwise. The i=i' condition is missing.
    
    So this gives us a tensor that is 1 when j=j' and k=k'', regardless of i and i'. That's not the multiplication tensor.
    
    I see the issue. The multiplication tensor M[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if i=i', j=j', k=k', and 0 otherwise.
    
    So we need the decomposition to also enforce i=i'.
    
    Let me try a different construction. For each (i,j) pair, define a term.
    
    Rank = n*m = 15.
    
    For t = (i,j) where i in [0,n), j in [0,m):
    
    U[i'*m+j', t] = delta_{i,i'} * delta_{j,j'}
    V[j'*p+k', t] = delta_{j,j'} * 1  (independent of k')
    W[k''*n+i'', t] = delta_{i,i''} * 1  (independent of k'')
    
    Then the product is:
    U[i*m+j, t] * V[j'*p+k, t] * W[k''*n+i', t]
    = delta_{i,i} * delta_{j,j} * delta_{j,j'} * delta_{i,i'}
    = delta_{j,j'} * delta_{i,i'}
    
    Summing over t=(i,j):
    sum_{i,j} delta_{j,j'} * delta_{i,i'} = 1 if i=i' and j=j', else 0
    
    But we also need k=k'. The k index is not being checked!
    
    So this gives us a tensor that is 1 when i=i' and j=j', regardless of k and k'. That's not the multiplication tensor either.
    
    I see. The problem is that each term can only "check" one index from each of the three tensors, but we need to check all three indices (i, j, k) simultaneously.
    
    The schoolbook decomposition uses one term per (i,j,k) triple, which is why it works.
    
    To get a lower rank, we need to find a way to check multiple (i,j,k) triples with a single term.
    
    This is possible if we use linear combinations. For example, Strassen's algorithm for 2x2 matrices uses 7 terms instead of 8, by finding clever linear combinations.
    
    For rectangular matrices, there are various constructions. Let me look for a known result.
    
    Actually, I recall that the rank of mu_{n,m,p} is at least max(n*p, m*p, n*m) and at most n*m*p. For n=3, m=5, p=7, the lower bound is 35.
    
    There's a construction by Pan or others that achieves rank = m * max(n, p) for certain dimensions.
    
    Let me try to construct a rank-35 decomposition.
    
    Idea: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can write this as: C = A * B = sum_j (A[:,j] * B[j,:])
    
    where A[:,j] is the j-th column of A (length n) and B[j,:] is the j-th row of B (length p).
    
    Each term A[:,j] * B[j,:] is an outer product, which is a rank-1 matrix.
    
    Now, to express this as a tensor decomposition, we need to write:
    sum_j (A[:,j] * B[j,:]) = sum_t (u_t^T A) * (v_t^T B) * w_t
    
    where u_t is in R^{n*m}, v_t is in R^{m*p}, w_t is in R^{n*p}.
    
    For the j-th term, we want:
    u_j^T A = A[:,j]  (the j-th column of A)
    v_j^T B = B[j,:]  (the j-th row of B)
    w_j = e_{i,k} for some i,k? No, that doesn't work.
    
    Actually, the outer product A[:,j] * B[j,:] is a matrix of size n x p, which can be written as:
    (A[:,j] * B[j,:])_{i,k} = A[i,j] * B[j,k]
    
    So the j-th term contributes A[i,j] * B[j,k] to C[i,k].
    
    To express this as a tensor term, we need:
    u_j[(i',j')] = delta_{j,j'}  (so that u_j^T A = sum_{i',j'} delta_{j,j'} A[i',j'] = sum_{i'} A[i',j] = A[:,j])
    
    Wait, u_j^T A should be a scalar, but A[:,j] is a vector. I'm confusing myself.
    
    Let me restart the tensor decomposition more carefully.
    
    The multiplication tensor M is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p}).
    
    A rank-1 tensor is of the form u ⊗ v ⊗ w where u in (R^{n*m})^*, v in (R^{m*p})^*, w in (R^{n*p}).
    
    The action of u ⊗ v ⊗ w on A in R^{n*m} and B in R^{m*p} is:
    (u ⊗ v ⊗ w)(A, B) = (u^T A) * (v^T B) * w
    
    This gives a scalar times w, which is a vector in R^{n*p}.
    
    For the multiplication, we want:
    M(A, B) = A * B  (the matrix product)
    
    So we need:
    sum_t (u_t^T A) * (v_t^T B) * w_t = A * B
    
    For the schoolbook decomposition, for each (i,j,k):
    u_{i,j,k}[(i',j')] = delta_{i,i'} * delta_{j,j'}
    v_{i,j,k}[(j',k')] = delta_{j,j'} * delta_{k,k'}
    w_{i,j,k}[(k'',i'')] = delta_{k,k''} * delta_{i,i''}
    
    Then (u_{i,j,k}^T A) = A[i,j]
    (v_{i,j,k}^T B) = B[j,k]
    w_{i,j,k} = e_{k,i}  (the standard basis vector in R^{n*p} corresponding to position (k,i))
    
    So the term is A[i,j] * B[j,k] * e_{k,i}, which contributes A[i,j]*B[j,k] to the (i,k) entry of the result.
    
    Summing over all (i,j,k) gives the full matrix product.
    
    Now, to get a lower rank, we need to find fewer terms that still give the correct result.
    
    One approach: use the fact that for fixed j, we have n*p terms (one for each (i,k)). Can we combine these?
    
    For fixed j, the contribution is:
    sum_{i,k} A[i,j] * B[j,k] * e_{k,i}
    = A[:,j] * B[j,:]  (the outer product)
    
    So for each j, we need to decompose the outer product A[:,j] * B[j,:] into a sum of rank-1 tensor terms.
    
    The outer product of two vectors (one of length n, one of length p) is a rank-1 matrix, but as a tensor in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p}), it's not necessarily rank 1.
    
    Actually, for fixed j, the term A[:,j] * B[j,:] can be written as:
    sum_{i,k} (delta_{j,j} A[i,j]) * (delta_{j,j} B[j,k]) * e_{k,i}
    = sum_{i,k} u_{i,j} * v_{j,k} * w_{k,i}
    
    where u_{i,j}[(i',j')] = delta_{i,i'} * delta_{j,j'}
    v_{j,k}[(j',k')] = delta_{j,j'} * delta_{k,k'}
    w_{k,i}[(k'',i'')] = delta_{k,k''} * delta_{i,i''}
    
    This is the schoolbook decomposition for fixed j, with n*p terms.
    
    To reduce the rank, we need to find a more efficient decomposition of the outer product.
    
    For the outer product of an n-vector and a p-vector, the minimum rank as a tensor in (R^n)^* ⊗ (R^p)^* ⊗ (R^{n*p}) is 1 (it's already rank 1).
    
    But in our case, the vectors are embedded in larger spaces (R^{n*m} and R^{m*p}), so the rank might be higher.
    
    I think I need to accept that for now, the schoolbook decomposition with rank 105 is the safest bet, and try to optimize it.
    
    However, the task says "rank-minimal", so I should try to find a lower rank.
    
    Let me try a different approach: use the fact that we can use linear combinations of the basis vectors.
    
    For example, for the 2x2 case, Strassen's algorithm uses 7 terms instead of 8.
    
    For the 3x5x7 case, let me try to find a construction with rank 35.
    
    Actually, I just realized that I can use the following construction:
    
    For each (j, k) pair, define a term that computes the contribution of B[j,k] to all C[i,k].
    
    For t = (j, k) where j in [0,m), k in [0,p):
    
    u_t[(i',j')] = delta_{j,j'}  (independent of i')
    v_t[(j',k')] = delta_{j,j'} * delta_{k,k'}
    w_t[(k'',i'')] = delta_{k,k''} * 1  (independent of i'')
    
    Then (u_t^T A) = sum_{i',j'} delta_{j,j'} A[i',j'] = sum_{i'} A[i',j] = A[:,j]  (a vector of length n)
    
    Wait, u_t^T A is a scalar, not a vector. I'm making the same mistake again.
    
    Let me be very careful. u_t is in (R^{n*m})^*, so u_t^T A is a scalar.
    
    So I can't have u_t^T A = A[:,j] because that's a vector.
    
    I need to rethink this.
    
    OK here's another idea. What if I use the following:
    
    For each i in [0,n), define a term that computes the i-th row of C.
    
    For t = i:
    u_t[(i',j')] = delta_{i,i'}  (independent of j')
    v_t[(j',k')] = 1  (all ones)
    w_t[(k'',i'')] = delta_{i,i''}  (independent of k'')
    
    Then (u_t^T A) = sum_{i',j'} delta_{i,i'} A[i',j'] = sum_{j'} A[i,j'] = sum of the i-th row of A
    
    That's not what we want.
    
    I give up trying to find a clever decomposition and will just use the schoolbook decomposition, but try to optimize the code.
    
    Actually, wait. Let me try one more thing.
    
    What if I use the following construction with rank = m * p = 35?
    
    For t = (j, k) where j in [0,m), k in [0,p):
    
    u_t[(i',j')] = delta_{j,j'}
    v_t[(j',k')] = delta_{j,j'} * delta_{k,k'}
    w_t[(k'',i'')] = delta_{k,k''}
    
    Then the product is:
    u_t[(i,j)] * v_t[(j',k')] * w_t[(k'',i'')]
    = delta_{j,j} * delta_{j,j'} * delta_{k,k'} * delta_{k,k''}
    = delta_{j,j'} * delta_{k,k''}
    
    Summing over t=(j,k):
    sum_{j,k} delta_{j,j'} * delta_{k,k''} = 1 if j=j' and k=k'', else 0
    
    But we need this to be 1 if i=i', j=j', k=k'', and 0 otherwise. The i=i' condition is still missing.
    
    So this doesn't work.
    
    I think the issue is that the multiplication tensor has a specific structure that requires checking all three indices (i, j, k), and the only way to do this with rank-1 tensors is to have one term per (i,j,k) triple, which gives rank n*m*p.
    
    Unless we use linear combinations, like in Strassen's algorithm.
    
    For now, I'll just use the schoolbook decomposition and return it.
    
    def construct():
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
    
    This is the same as the original program. The score will be 1.0.
    
    But the task says to find a "rank-minimal" decomposition, so I should try harder.
    
    Let me look for known results on the rank of rectangular matrix multiplication tensors.
    
    After some thought, I believe that for n=3, m=5, p=7, the minimum rank is 35, and there is a construction that achieves this.
    
    The construction is based on the following idea:
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C = A * B = sum_j (A[:,j] ⊗ B[j,:])
    
    where A[:,j] is the j-th column of A and B[j,:] is the j-th row of B.
    
    Now, A[:,j] ⊗ B[j,:] is an n x p matrix, which is the outer product of an n-vector and a p-vector.
    
    To express this as a tensor decomposition, we need to write:
    sum_j (A[:,j] ⊗ B[j,:]) = sum_t (u_t^T A) * (v_t^T B) * w_t
    
    For the j-th term, we want:
    (u_j^T A) * (v_j^T B) * w_j = A[:,j] ⊗ B[j,:]
    
    This is not possible with a single term because (u_j^T A) and (v_j^T B) are scalars, and w_j is a vector, so the product is a scalar times a vector, which is a vector, not a matrix.
    
    I see the fundamental issue: the tensor decomposition gives a vector in R^{n*p}, not a matrix. The matrix is just a reshaping of this vector.
    
    So the decomposition is:
    vec(C) = sum_t (u_t^T vec(A)) * (v_t^T vec(B)) * w_t
    
    where vec() is the vectorization operation.
    
    For the schoolbook decomposition, this works with rank n*m*p.
    
    I think for now, I'll just use the schoolbook decomposition and accept the score of 1.0.
    
    Actually, let me try one more thing. What if I use the following construction:
    
    For each (i, j) pair, define a term that computes the contribution of A[i,j] to all C[i,k].
    
    For t = (i, j) where i in [0,n), j in [0,m):
    
    u_t[(i',j')] = delta_{i,i'} * delta_{j,j'}
    v_t[(j',k')] = delta_{j,j'}  (independent of k')
    w_t[(k'',i'')] = delta_{i,i''}  (independent of k'')
    
    Then the product is:
    u_t[(i,j)] * v_t[(j',k')] * w_t[(k'',i'')]
    = delta_{i,i} * delta_{j,j} * delta_{j,j'} * delta_{i,i''}
    = delta_{j,j'} * delta_{i,i''}
    
    Summing over t=(i,j):
    sum_{i,j} delta_{j,j'} * delta_{i,i''} = 1 if i=i'' and j=j', else 0
    
    But we need this to be 1 if i=i', j=j', k=k'', and 0 otherwise. The k=k'' condition is missing.
    
    So this doesn't work either.
    
    I'm stuck. Let me just use the schoolbook decomposition.
    """
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
