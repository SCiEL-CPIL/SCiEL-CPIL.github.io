"""Matrix multiplication 3x5x5: structured Kronecker product decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix using a structured Kronecker product approach.
    
    We express the matrix multiplication tensor T_{i,j,k} = delta_{(i,j),(j',k)} 
    where the output index is (k, i) following W[k*n+i, r] convention.
    
    The key insight: we can decompose using the separability of the outer product.
    The 3x5 matrix A has entries A[i,j] and the 5x5 matrix B has entries B[j,k].
    The product C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor is T[i,k,j] = A[i,j]*B[j,k] which factors as a sum over j of
    rank-1 tensors: for each j, the tensor is u_j ⊗ v_j ⊗ w_j where
    u_j has 1 at position i (for all i, it's actually the identity in i for fixed j... 
    
    Wait, let me think more carefully. The matrix multiplication tensor for n×m × m×p → n×p
    has entries T_{i,k,j} = δ_{i,i}δ_{k,k}... no.
    
    Actually, the bilinear map is: (A, B) → C where C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor T has indices (i, k, j) with T[i,k,j] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    
    So T is a tensor of shape (n*p, m) where for each output (i,k) and input pair (j), 
    T[(i,k), j] = 1. But that's not quite right for a tensor decomposition.
    
    Let me reconsider. The matrix multiplication tensor T ∈ R^{(n*p) × (n*m) × (m*p)} is defined by:
    T[(i,k), (i,j), (j,k)] = 1 for all valid i, j, k, and 0 otherwise.
    
    Wait, that's the standard definition. For each triple (i,j,k), there's a 1 in T at position
    ((i,k), (i,j), (j,k)).
    
    So the tensor has rank at most n*m*p (schoolbook), but we want to find a lower rank decomposition.
    
    For n=3, m=5, p=5: schoolbook rank = 75.
    
    The directive suggests using a Kronecker product structure. Let's think about this differently.
    
    We can write the matrix multiplication as:
    C = A * B where A is 3x5, B is 5x5, C is 3x5.
    
    One approach: view A as a 3x1 block matrix where each block is 1x5, and B as a 5x1 block matrix 
    where each block is 1x5. Then C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor T_{(i,k), (i,j), (j,k)} = 1.
    
    For a Kronecker product decomposition, we can try to express this as:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r ∈ R^{n*p}, v_r ∈ R^{n*m}, w_r ∈ R^{m*p}.
    
    The directive suggests leveraging the separability. Let's consider that the index j couples
    the row index of A with the column index of B. 
    
    A structured approach: for each j in {0,...,m-1}, we have a "slice" of the tensor. 
    Within each slice, the tensor is:
    T_slice_j[(i,k), (i,j), (j,k)] = 1
    
    This slice itself can be written as a sum of rank-1 terms. Specifically, for fixed j:
    T_slice_j = sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_{(i,k)} ⊗ e_{(i,j)} ⊗ e_{(j,k)}
    
    So each slice has rank n*p, and total rank is m*n*p = 75.
    
    But we can do better by finding linear dependencies across slices.
    
    The Kronecker product idea: if we could separate the i-index from the j-index and k-index,
    we might reduce the rank. 
    
    Actually, let me think about this problem from a different angle. The matrix multiplication
    tensor for (n,m,p) has known lower bounds. For (3,5,5), the minimum rank is not trivially known.
    
    However, the directive specifically says to use a Kronecker product decomposition that gives
    rank 15 (3*5). Let's see if that's possible.
    
    If we can achieve rank 15, that would be a huge improvement over 75.
    
    Let's try to construct such a decomposition. The idea is to use the fact that the matrix
    multiplication can be viewed as a tensor product of simpler operations.
    
    Consider: C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we fix i and k, this is a bilinear form in the entries of A and B. 
    
    For a rank-r decomposition, we need r triples of vectors (u_r, v_r, w_r) such that:
    sum_r u_r[i,k] * v_r[i,j] * w_r[j,k] = delta_{i,i}delta_{k,k}... 
    
    No, the correct condition is:
    sum_r U[(i,k), r] * V[(i,j), r] * W[(j,k), r] = 1 for all i,j,k
    and 0 otherwise (but since each (i,k) pair only appears with specific (i,j) and (j,k) pairs,
    we just need the sum to equal 1 when the indices are consistent).
    
    Actually, the tensor T is defined such that T[(i,k), (i,j), (j,k)] = 1 and T is zero elsewhere.
    So we need:
    sum_r U[(i,k), r] * V[(i,j), r] * W[(j,k), r] = 1 if the indices are "compatible" (i.e., the j in V matches the j in W, and the i in U matches the i in V, and the k in U matches the k in W)
    
    More precisely, for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}:
    sum_r U[(i,k), r] * V[(i,j), r] * W[(j,k), r] = 1
    
    And for all other index combinations (where the indices don't form a valid triple), the sum should be 0.
    But actually, the tensor is only non-zero at valid triples, so we only need the condition above.
    
    Now, can we find a rank-15 decomposition? That would require 15 vectors U_r, V_r, W_r.
    
    The directive suggests using a Kronecker product structure. Let's try:
    U[(i,k), r] = a_r[i] * b_r[k]
    V[(i,j), r] = c_r[i] * d_r[j]  
    W[(j,k), r] = e_r[j] * f_r[k]
    
    Then the product is:
    sum_r a_r[i] * b_r[k] * c_r[i] * d_r[j] * e_r[j] * f_r[k]
    = sum_r (a_r[i]*c_r[i]) * (d_r[j]*e_r[j]) * (b_r[k]*f_r[k])
    
    For this to equal 1 for all i,j,k, we need:
    sum_r (a_r[i]*c_r[i]) * (d_r[j]*e_r[j]) * (b_r[k]*f_r[k]) = 1
    
    This can be written as:
    Let A_r[i] = a_r[i]*c_r[i], D_r[j] = d_r[j]*e_r[j], B_r[k] = b_r[k]*f_r[k]
    Then sum_r A_r[i] * D_r[j] * B_r[k] = 1
    
    This is a rank-r decomposition of the all-ones tensor, which is impossible for r < min(n,m,p) unless we allow the decomposition to be more complex.
    
    Wait, that's not right. The all-ones tensor has rank 1 (it's just the outer product of three all-ones vectors). So if we could make A_r[i]*D_r[j]*B_r[k] sum to 1, we'd need at least 1 term, but the issue is that the indices i, j, k are coupled in the original problem.
    
    I think I'm overcomplicating this. Let me re-read the directive.
    
    The directive says: "leveraging the separability of the outer product structure to reduce the search space from the full 75-dimensional tensor to a rank-15 decomposition derived from the tensor product of a rank-3 identity and a rank-5 identity."
    
    So the idea is: rank = 3 * 5 = 15.
    
    Let me try a different structured approach. Suppose we index the rank-1 terms by (a, b) where a in {0,1,2} and b in {0,1,2,3,4}.
    
    For term (a,b), define:
    U[(i,k), (a,b)] = delta_{i,a} * something involving k
    V[(i,j), (a,b)] = delta_{i,a} * something involving j  
    W[(j,k), (a,b)] = something involving j and k
    
    Let's try:
    U[(i,k), (a,b)] = delta_{i,a} * delta_{k,b}
    V[(i,j), (a,b)] = delta_{i,a} * delta_{j,b}
    W[(j,k), (a,b)] = delta_{j,b} * delta_{k,b}
    
    Then the sum over (a,b) of U*V*W for fixed (i,k,j):
    sum_{a,b} delta_{i,a}*delta_{k,b} * delta_{i,a}*delta_{j,b} * delta_{j,b}*delta_{k,b}
    = sum_{a,b} delta_{i,a}^2 * delta_{k,b}^2 * delta_{j,b}^2
    = delta_{i,i} * delta_{k,k} * delta_{j,j} ... no, this doesn't work.
    
    Let me try yet another approach. 
    
    Actually, I realize I should just implement the schoolbook decomposition first to ensure correctness, and then try to optimize. But the task asks for a rank-minimal decomposition, and the current score is 1.0 (rank 75). Any improvement would be better.
    
    Let me think about whether rank 15 is actually achievable. For matrix multiplication (n,m,p), the minimum rank is known to be at least max(n*m, m*p, n*p) for some cases, but this is not a general lower bound.
    
    For (3,5,5), n*p = 15, m*p = 25, n*m = 15. So the lower bound from this heuristic is 25.
    
    Hmm, so rank 15 might not be achievable. Let me reconsider.
    
    Actually, I think the directive might be suggesting a different kind of decomposition. Let me just implement a reasonable structured decomposition and see what rank we get.
    
    Let me go with the schoolbook approach but structure it clearly, since I'm not confident about achieving a lower rank with the given constraints.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * p)]
    V = [[0] * rank for _ in range(n * m)]
    W = [[0] * rank for _ in range(m * p)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * p + k][column] = 1
                V[i * m + j][column] = 1
                W[j * p + k][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
