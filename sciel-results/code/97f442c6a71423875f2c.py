"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    The tensor T has indices (i, j, k) with i in [0,3), j in [0,5), k in [0,5).
    T[i,j,k] = 1 if the output element (i,k) depends on A[i,j]*B[j,k].
    
    We represent this as a bilinear map:
    U: (n*m) x rank  -- A indices (i,j)
    V: (m*p) x rank  -- B indices (j,k)
    W: (p*n) x rank  -- C indices (k,n+i) per trace convention W[k*n+i, r]
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*m+j, r] * B[j,k] ... wait, let me be more precise.
    
    Actually, the standard formulation:
    For each rank-r term, we have:
      A contributes via U[i*m+j, r]
      B contributes via V[j*p+k, r]
      C receives via W[k*n+i, r]
    
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 if (i,j,k) is a valid triple in the multiplication tensor, else 0.
    
    For 3x5*5x5, the valid triples are all (i,j,k) with i in [0,3), j in [0,5), k in [0,5),
    so T[i,j,k] = 1 for all such triples. Wait, that's not right either.
    
    Let me reconsider. The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is the full tensor of ones on the index set [0,n) x [0,m) x [0,p).
    
    The rank of this tensor is the minimum number of rank-1 tensors needed to sum to it.
    A rank-1 tensor is u_i * v_j * w_k.
    
    So we need: sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    This is equivalent to finding matrices U (n x R), V (m x R), W (p x R) such that
    for all i,j,k: sum_r U[i,r] * V[j,r] * W[k,r] = 1.
    
    This can be rewritten as: for all i,k: sum_r U[i,r] * W[k,r] * (sum_j V[j,r]) = 1? No.
    
    Actually, fix i and k: sum_j sum_r U[i,r] V[j,r] W[k,r] = sum_r U[i,r] W[k,r] (sum_j V[j,r]) = 1.
    
    Let S[r] = sum_j V[j,r]. Then we need: sum_r U[i,r] * S[r] * W[k,r] = 1 for all i,k.
    
    This means the matrix M[i,k] = sum_r U[i,r] * S[r] * W[k,r] must be the all-ones matrix.
    
    Equivalently, let A' = U * diag(S) (i.e., A'[i,r] = U[i,r] * S[r]), then M = A' * W^T must be all-ones.
    
    The all-ones matrix of size n x p has rank 1 (if n,p >= 1). So we can achieve this with rank 1!
    
    Wait, that seems too good. Let me check: if R=1, U[i,0]=1, V[j,0]=1, W[k,0]=1 for all i,j,k.
    Then sum_r U[i,r] V[j,r] W[k,r] = 1*1*1 = 1. Yes!
    
    But wait, that would mean the rank of the multiplication tensor for ANY n,m,p is 1, which is clearly wrong.
    
    I'm confusing the tensor. The matrix multiplication tensor is NOT the all-ones tensor. Let me reconsider.
    
    The matrix multiplication tensor T_{n,m,p} is defined as:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    where e_i, e_j, e_k are basis vectors. This IS the all-ones tensor in the standard basis.
    
    The rank of this tensor is indeed the minimum R such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    
    And I showed that R=1 works: take u_r = (1,...,1), v_r = (1,...,1), w_r = (1,...,1).
    
    But this contradicts known results. The rank of the 2x2x2 matrix multiplication tensor is 7, not 1.
    
    I see my error. The tensor T is not sum of e_i ⊗ e_j ⊗ e_k. Let me look this up properly.
    
    The matrix multiplication tensor is:
    T_{n,m,p} = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i ⊗ f_j ⊗ g_{k,i}
    
    where g_{k,i} is the basis vector in the output space corresponding to C[i,k].
    
    So the output space is indexed by (i,k), and the tensor has T[i,j,(i,k)] = 1 and 0 otherwise.
    
    In the flattened output index (let's say index = k*n + i per the problem's convention W[k*n+i, r]),
    the tensor entry at position (i*m+j, j*p+k, k*n+i) is 1, and all other entries are 0.
    
    So the tensor is:
    T[A_idx, B_idx, C_idx] = 1 if A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i for some i,j,k, else 0.
    
    This is a 0-1 tensor with specific structure.
    
    For the decomposition, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) valid, else 0.
    
    Since every (i,j,k) with i in [0,3), j in [0,5), k in [0,5) is valid, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,3), j in [0,5), k in [0,5).
    
    And for invalid triples (which don't exist in this case since all combos are valid), it should be 0.
    
    Wait, I think I was right initially. Every combination of i, j, k is a valid term in the sum.
    So the tensor IS the all-ones tensor on the product space.
    
    But then why is the rank of 2x2x2 matmul 7 and not 1?
    
    OH. I see. The tensor is not "sum over all i,j,k of e_i ⊗ e_j ⊗ e_k". The tensor represents the bilinear map, and its rank is the bilinear complexity. The tensor entries are T[i,j,k] where i indexes A, j indexes B, k indexes C. But the constraint is that C[i,k] = sum_j A[i,j] B[j,k], so the tensor has non-zero entries only where the indices are compatible.
    
    For a fixed output position (i,k), the relevant input indices are (i,j) for A and (j,k) for B, summed over j. So the tensor entry T[(i,j), (j,k), (i,k)] = 1.
    
    In the flattened indexing:
    A index: i*m + j
    B index: j*p + k  
    C index: k*n + i (per problem convention)
    
    So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    
    For n=3, m=5, p=5:
    A has 15 entries, B has 25 entries, C has 15 entries.
    The tensor has 15*25*15 = 5625 entries, most of which are 0.
    
    The non-zero entries are: for each i in [0,3), j in [0,5), k in [0,5):
    T[i*5+j, j*5+k, k*3+i] = 1.
    That's 3*5*5 = 75 non-zero entries.
    
    Now, the rank is the minimum R such that we can write T as sum of R rank-1 tensors.
    
    The schoolbook solution uses R = 75, with each rank-1 tensor being a single non-zero entry.
    
    Can we do better? Yes, by sharing rank-1 terms across multiple (i,j,k) triples.
    
    For example, if we fix j, then for all i and k, we have T[i*5+j, j*5+k, k*3+i] = 1.
    This sub-tensor (for fixed j) has 3*5 = 15 non-zero entries. Can we decompose this with fewer than 15 rank-1 terms?
    
    The sub-tensor for fixed j is: sum_{i=0}^{2} sum_{k=0}^{4} e_{i*5+j} ⊗ e_{j*5+k} ⊗ e_{k*3+i}
    
    This is a 3x5x5 tensor (in the i,k dimensions, with j fixed). Its rank is at most 15 (schoolbook for this sub-problem).
    
    But we can potentially share across different j values.
    
    Let me think about this differently. The total tensor is:
    T = sum_{j=0}^{4} T_j
    
    where T_j = sum_{i=0}^{2} sum_{k=0}^{4} e_{i*5+j} ⊗ e_{j*5+k} ⊗ e_{k*3+i}
    
    Each T_j lives in a different "slice" of the A and B spaces (different j for A's column and B's row), but they all contribute to the same C space.
    
    The key insight: for a given (i,k) pair in C, the contributions come from all j:
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    So the tensor entry for output (i,k) involves all 5 j values.
    
    To minimize rank, we want to find U, V, W with as few columns as possible such that:
    sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all i,j,k.
    
    This is a system of 75 equations (one for each (i,j,k)) with 3*5*R + 5*5*R + 5*3*R = 75R unknowns (entries of U, V, W).
    
    For R=19, we have 1425 unknowns and 75 equations, so it's very underdetermined. The question is whether a solution exists with the additional constraint that the decomposition is exact.
    
    Let me try a constructive approach. We'll use the fact that the problem has structure.
    
    One known technique: for matmul(n,m,p), if we can find a decomposition of rank R, then we can often improve by exploiting symmetries or using known lower bounds.
    
    For 3x5x5, the schoolbook rank is 75. The trivial lower bound is max(n*p, m*p, n*m) = max(15, 25, 15) = 25.
    
    Actually, a better lower bound: the rank is at least the max of the ranks of any "fiber" of the tensor.
    
    Let me just try to construct a solution with rank 19 or lower using a search or algebraic method.
    
    Given the complexity, let me try a simple heuristic: start with the schoolbook solution and try to merge terms.
    
    Actually, let me just return a solution with rank 25 (the trivial lower bound) if I can construct one, or fall back to 75.
    
    Let me try rank 25:
    We need U (15 x 25), V (25 x 25), W (15 x 25).
    
    Idea: for each j in [0,5), we have a "block" of 3*5=15 terms (for all i,k). If we could decompose each block with rank 5, total rank would be 25.
    
    For fixed j, the sub-tensor is: sum_{i,k} e_{i*5+j} ⊗ e_{j*5+k} ⊗ e_{k*3+i}
    
    Can we write this as sum_{r=1}^{5} u_r ⊗ v_r ⊗ w_r?
    
    This requires: sum_{r=1}^{5} u_r[i] * v_r[k] * w_r[i,k] = 1 for all i in [0,3), k in [0,5).
    
    (Here u_r is a 3-vector, v_r is a 5-vector, w_r is a 3x5 matrix... wait, w_r should be a vector in the C space, which has 15 entries, but for fixed j, only 15 entries are relevant, indexed by (i,k) -> k*3+i.)
    
    Actually, for fixed j, the C indices used are k*3+i for i in [0,3), k in [0,5), which is all 15 C indices.
    
    So for fixed j, we need: sum_{r=1}^{5} u_r[i] * v_r[k] * w_r[k*3+i] = 1 for all i,k.
    
    This is: sum_{r=1}^{5} u_r[i] * w_r[k*3+i] * v_r[k] = 1.
    
    Let M_r[i,k] = u_r[i] * w_r[k*3+i]. Then we need sum_{r=1}^{5} M_r[i,k] * v_r[k] = 1 for all i,k.
    
    This means: for each k, sum_{r=1}^{5} M_r[i,k] * v_r[k] = 1 for all i.
    
    So the vector (sum_r M_r[i,k] v_r[k])_i must be (1,1,1) for each k.
    
    This is a system that might have a solution. Let me try to construct one.
    
    This is getting complex. Let me just implement a solution that uses rank 75 (schoolbook) for now, but structure the code to allow easy experimentation.
    
    Actually, let me try a different approach: use known results or a simple construction.
    
    For now, I'll return the schoolbook solution with rank 75, but I'll add a TODO to improve it.
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
