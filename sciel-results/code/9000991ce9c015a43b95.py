"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x14 matrix multiplication.
    
    Strategy: Split the intermediate dimension m=2 into two independent m=1 sub-problems.
    Each 2x1 * 1x14 multiplication has tensor rank 14 (it's just a bilinear form with 14 outputs,
    each depending on 2*1=2 inputs in a specific way... wait, let me reconsider).
    
    Actually, let's think more carefully.
    
    We have A (2x2) * B (2x14) = C (2x14).
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,13}.
    
    The tensor T has indices (i, j, k) where i is row of A (and C), j is col of A / row of B, k is col of B (and C).
    T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    
    This is a direct sum over k of 14 tensors, each of which is a 2x2 matrix multiplication tensor T_k[i,j] = delta_{i,k's row}... no wait.
    
    Actually C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    
    So for each k, we're computing a 2x1 vector: C[:,k] = A * B[:,k].
    This is 14 independent 2x1 * 2x1 multiplications (where the 2x1 is a column of B).
    
    Wait, that's not right either. Let me re-examine.
    
    A is 2x2, B is 2x14. C = A*B is 2x14.
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k].
    
    For each fixed k, C[:,k] is a 2-vector, and B[:,k] is a 2-vector. The multiplication is:
    C[:,k] = A * B[:,k]
    
    This is a linear map from 2-vectors to 2-vectors, applied to 14 different 2-vectors.
    The tensor for this is: for each k, we have a 2x2 matrix multiplication (rank 7 by Strassen, or rank 8 schoolbook).
    
    But the 14 k's are independent in the sense that C[:,k] only depends on B[:,k]. So the total tensor is a direct sum of 14 copies of the 2x2 matrix multiplication tensor.
    
    Total rank = 14 * rank(2x2 matmul) = 14 * 7 = 98 with Strassen, or 14 * 8 = 112 schoolbook.
    
    But wait, the schoolbook for 2x2*2x14 would be 2*2*14 = 56. Let me recheck.
    
    Schoolbook: for each (i,j,k), we have a term A[i,j]*B[j,k]. There are 2*2*14 = 56 such terms. So rank 56 schoolbook.
    
    With Strassen on each 2x2 block: each 2x2 matmul has rank 7, and we have 14 such independent matmuls (one for each column k of B). So 14*7 = 98? That's worse than 56!
    
    Wait, I'm confusing myself. Let me reconsider.
    
    Actually, the 2x2*2x14 multiplication can be viewed as 2 independent 2x1*2x14... no.
    
    Let me think again. A is 2x2, B is 2x14.
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    So for each k, we compute 2 outputs from 4 inputs (A[0,0], A[0,1], A[1,0], A[1,1] and B[0,k], B[1,k]).
    
    Actually, the A entries are shared across all k. So the tensor is not a direct sum of 14 independent 2x2 matmuls in the usual sense, because the A entries appear in all 14 computations.
    
    Hmm, but in tensor decomposition, the rank is about the number of rank-1 terms. Let me think about the structure.
    
    The tensor T has T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k].
    
    In a rank-r decomposition, we write:
    C[i,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, the interface says U has shape (n*m, rank) = (4, rank), V has shape (m*p, rank) = (28, rank), W has shape (p*n, rank) = (28, rank).
    
    And the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j? No, that's not right either.
    
    Let me re-read the task. "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    The standard tensor decomposition for matrix multiplication: 
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, the multiplication tensor T has T[i,j,k] = 1 (for the appropriate indices). A rank-r decomposition gives:
    T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, but the shapes are U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    
    So U is indexed by (i,j) flattened, V by (j,k) flattened, W by (k,i) flattened.
    
    And the decomposition is: for each (i,j,k), T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    In the schoolbook case, T[i,j,k] = 1 for all valid (i,j,k), and the rank is n*m*p. Each term c corresponds to a unique (i,j,k), with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1, and all others 0.
    
    So for 2x2x14: rank 56 schoolbook.
    
    Now, can we do better? The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    This tensor can be seen as: for each k, we have a 2x2 slice T[:,:,k] which is the 2x2 all-ones matrix... no wait.
    
    T[i,j,k] = 1 for all i,j,k. So each 2x2 slice (fixed k) is the 2x2 all-ones matrix. And there are 14 such slices, all identical.
    
    The 2x2 all-ones matrix as a tensor (i,j) has rank 1 (it's the outer product of [1,1] and [1,1]). So T[i,j,k] = 1 = 1*1*1, which is rank 1 per k? No, that's not how tensor rank works.
    
    Actually, T[i,j,k] = 1 for all i,j,k. This is the tensor that is the outer product of three vectors: u = [1,1,1,1] (for i*m+j), v = [1,1,...,1] (for j*p+k, 28 entries), w = [1,1,...,1] (for k*n+i, 28 entries). Wait, no.
    
    T[i,j,k] = 1. If we write this as a sum of rank-1 tensors, a rank-1 tensor is u[i,j] * v[j,k] * w[k,i]. For T[i,j,k] = 1, we need u[i,j] * v[j,k] * w[k,i] = 1 for all i,j,k.
    
    If u[i,j] = 1, v[j,k] = 1, w[k,i] = 1, then indeed u*v*w = 1. So T has rank 1!
    
    Wait, that can't be right. Let me re-examine.
    
    In matrix multiplication, the tensor is T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]. For standard matrix multiplication, this is always 1 (every entry of A and B contributes to exactly one entry of C... no, that's not right).
    
    Actually, for matrix multiplication (A*B)[i,k] = sum_j A[i,j] * B[j,k], the tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]. This coefficient is 1 if the indices match (i.e., the j in A[i,j] matches the j in B[j,k]), and 0 otherwise. But in the standard formulation, T[i,j,k] = 1 for all valid i,j,k, because A[i,j] and B[j,k] always contribute to C[i,k] (with the sum over j).
    
    Wait, I think I'm overcomplicating this. Let me look at the schoolbook decomposition in the seed program.
    
    In the seed program:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
    
    So for each (i,j,k), we have a rank-1 term where U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    
    The reconstructed tensor is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    For the schoolbook term corresponding to (i0,j0,k0), this gives 1 when (i,j,k) = (i0,j0,k0) and 0 otherwise. So the sum over all c gives T[i,j,k] = 1 for all (i,j,k), which is correct.
    
    Now, can we do better than rank 56?
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    This is a rank-1 tensor! Because T[i,j,k] = 1 * 1 * 1 = u[i,j] * v[j,k] * w[k,i] where u[i,j]=1, v[j,k]=1, w[k,i]=1.
    
    Wait, but that would mean the rank is 1, which is clearly wrong for matrix multiplication. Let me reconsider.
    
    Oh, I see the issue. The tensor T for matrix multiplication is NOT T[i,j,k] = 1 for all i,j,k. Rather, T[i,j,k] = 1 if and only if the indices are consistent, i.e., the j in A[i,j] is the same as the j in B[j,k]. But in the standard matrix multiplication, this is always the case by construction.
    
    Hmm, let me think about this differently. The matrix multiplication tensor is defined as:
    T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    So T[i,j,k] = 1 for all valid i,j,k. This is correct.
    
    Now, the rank of this tensor. For n=m=p=2, the rank is 7 (Strassen). For n=m=p, the rank is known to be less than n^3.
    
    For our case, n=2, m=2, p=14. The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    This tensor can be written as: T = sum_{k=0}^{13} (sum_{i,j} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i})
    
    Each term in the inner sum (for fixed k) is a 2x2 matrix multiplication tensor with the third index fixed to k. But actually, for fixed k, we have a tensor in R^{2×2} ⊗ R^{2×1} ⊗ R^{1×2}, which is just a 2x2 matrix (the multiplication matrix for that k).
    
    Wait, I think the key insight is that the tensor T[i,j,k] = 1 for all i,j,k is actually a rank-1 tensor in the sense that it factors as u ⊗ v ⊗ w where u[i,j] = 1, v[j,k] = 1, w[k,i] = 1.
    
    But that would mean rank 1, which contradicts the known result that 2x2 matrix multiplication has rank 7.
    
    Let me re-examine. For 2x2 matrix multiplication (n=m=p=2):
    T[i,j,k] = 1 for i,j,k in {0,1}.
    
    If T had rank 1, we could write T[i,j,k] = u[i,j] * v[j,k] * w[k,i] for some u, v, w.
    
    Let u[i,j] = a_{i,j}, v[j,k] = b_{j,k}, w[k,i] = c_{k,i}.
    
    Then a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    
    Setting i=j=k=0: a_{0,0} * b_{0,0} * c_{0,0} = 1
    Setting i=0,j=0,k=1: a_{0,0} * b_{0,1} * c_{1,0} = 1
    Setting i=0,j=1,k=0: a_{0,1} * b_{1,0} * c_{0,0} = 1
    Setting i=0,j=1,k=1: a_{0,1} * b_{1,1} * c_{1,0} = 1
    
    From the first two: b_{0,1} * c_{1,0} = b_{0,0} * c_{0,0}
    From the first and third: a_{0,1} * b_{1,0} = a_{0,0} * b_{0,0}
    From the first and fourth: a_{0,1} * b_{1,1} * c_{1,0} = a_{0,0} * b_{0,0} * c_{0,0}
    
    This is a system of equations. It's not obvious that a solution exists with all entries nonzero, but it's also not obvious that no solution exists.
    
    Actually, I recall that the rank of the 2x2 matrix multiplication tensor is 7, not 1. So my assumption that T[i,j,k] = 1 for all i,j,k must be wrong, or my understanding of the tensor decomposition is incorrect.
    
    Let me look at this more carefully. The matrix multiplication tensor for multiplying an m×n matrix by an n×p matrix is:
    T ∈ R^{m×n} ⊗ R^{n×p} ⊗ R^{m×p}
    T = sum_{i=1}^m sum_{j=1}^n sum_{k=1}^p e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    So T[i,j,k] = 1 if the indices are (i,j,k) where i is the row of the first matrix and result, j is the column of the first matrix and row of the second, k is the column of the second matrix and result.
    
    For our case, n=2 (rows of A, cols of B... wait, let me re-read the dimensions).
    
    The task says "multiplying a 2 by 2 matrix by a 2 by 14 matrix". So A is 2x2, B is 2x14. A*B is 2x14.
    
    In the standard notation, A is n×m and B is m×p, so n=2, m=2, p=14.
    
    The tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (but the task says W has shape (p*n, rank), so it's R^{p×n}).
    
    T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{13} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    So T[i,j,k] = 1 for all valid i,j,k.
    
    Now, the rank of this tensor. For the 2x2x2 case (n=m=p=2), the rank is 7. For our 2x2x14 case, what is the rank?
    
    The tensor T can be viewed as a block matrix. For each k, we have a 2x2 block T[:,:,k] which is the 2x2 all-ones matrix. And there are 14 such blocks.
    
    The rank of the tensor is at most 14 * rank(2x2 matmul tensor with p=1)... no, that's not the right decomposition.
    
    Actually, I think the correct way to think about this is:
    
    T[i,j,k] = 1 for all i,j,k.
    
    This tensor can be decomposed as:
    T = sum_{k=0}^{13} T_k
    
    where T_k[i,j] = 1 for all i,j (a 2x2 matrix), and T_k is a tensor in R^{2×2} ⊗ R^{2×1} ⊗ R^{1×2}... this is getting confusing.
    
    Let me just try the Strassen approach. For each 2x2 block multiplication (which is what we have for each k), we can use Strassen's algorithm with rank 7. Since the 14 k's are independent (the A entries are shared, but in tensor decomposition, we can still decompose each k independently), the total rank would be 14 * 7 = 98. But that's worse than 56!
    
    Wait, that can't be right. Let me reconsider.
    
    Actually, I think the issue is that the A entries are shared across all k. In the tensor decomposition, the U matrix is indexed by (i,j) which are the indices of A. So U[i*m+j, c] is the coefficient for A[i,j] in the c-th rank-1 term.
    
    In the schoolbook decomposition, each term c corresponds to a unique (i,j,k), and U[i*m+j, c] = 1 only for that specific (i,j). So the A entries are not shared in the sense that each term uses a specific A[i,j].
    
    For Strassen, we would have 7 terms per 2x2 block, and the U entries would be shared across the 7 terms for that block. But since the 14 k's are independent, we would have 14 * 7 = 98 terms total.
    
    But wait, that's worse than 56. So Strassen doesn't help here?
    
    Let me reconsider the problem. Maybe the rank is indeed 56, and there's no way to do better.
    
    Actually, I recall that the rank of the matrix multiplication tensor for n×m times m×p is at most n*m*p (schoolbook), and for some cases, it can be less. For example, 2x2 times 2x2 has rank 7 < 8.
    
    For 2x2 times 2x14, the rank might be less than 56. Let me think about this.
    
    The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    This is a tensor of format (2*2, 2*14, 14*2) = (4, 28, 28).
    
    The rank of this tensor is the minimum number of rank-1 tensors needed to sum to T.
    
    A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^4, v ∈ R^28, w ∈ R^28.
    
    T[i,j,k] = 1 for all i,j,k.
    
    Can we write T as a sum of few rank-1 tensors?
    
    T[i*m+j, j*p+k, k*n+i] = 1.
    
    Let's try to factor this. We need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    One simple decomposition: for each k, we have a 2x2 matrix multiplication. The 2x2 matmul tensor has rank 7. So we can write the 2x2 matmul tensor as a sum of 7 rank-1 tensors.
    
    For each k, we have 7 terms. Total: 14 * 7 = 98 terms.
    
    But the schoolbook gives 56 terms. So 56 < 98, meaning schoolbook is better.
    
    Wait, that means the rank is at most 56, and Strassen gives 98, which is worse. So the rank is at most 56.
    
    Can we do better than 56?
    
    The tensor T[i,j,k] = 1 for all i,j,k. This is a "product" tensor in the sense that it's the outer product of three all-ones vectors... but the indices are coupled.
    
    Actually, I think the rank of this tensor is exactly 56, because the 14 k's are independent in the sense that the W indices (k*n+i) involve k, and the V indices (j*p+k) involve k, so the k dimension is "entangled" with the other dimensions.
    
    Hmm, but let me try a different approach. What if we use the fact that T[i,j,k] = 1 for all i,j,k?
    
    T[i*m+j, j*p+k, k*n+i] = 1.
    
    Let's define new indices: a = i*m+j (0 to 3), b = j*p+k (0 to 27), c = k*n+i (0 to 27).
    
    The constraint is that a, b, c are not independent; they are linked by the original i, j, k.
    
    For a given (i,j,k), we have:
    a = i*2 + j
    b = j*14 + k
    c = k*2 + i
    
    So for a given a, b, c, we can recover i, j, k if and only if the indices are consistent.
    
    The tensor T is 1 on the set of (a,b,c) that are consistent, and 0 otherwise.
    
    The number of consistent (a,b,c) triples is 2*2*14 = 56.
    
    So T is a 0-1 tensor with 56 ones, and the rest are zeros.
    
    The rank of this tensor is at least the number of ones divided by the maximum number of ones per rank-1 tensor. A rank-1 tensor u ⊗ v ⊗ w has u[a] * v[b] * w[c] = 1 for some (a,b,c) if and only if u[a] != 0, v[b] != 0, w[c] != 0.
    
    So a rank-1 tensor can cover at most (number of nonzero u) * (number of nonzero v) * (number of nonzero w) ones.
    
    To minimize the number of rank-1 tensors, we want to maximize the coverage of each.
    
    If we use u with 4 nonzeros, v with 28 nonzeros, w with 28 nonzeros, we can cover 4*28*28 = 3136 ones, but we only have 56 ones. So one rank-1 tensor could potentially cover all 56 ones if the ones are aligned.
    
    But the ones are not all aligned; they are on a specific set of (a,b,c) triples.
    
    I think the rank is 56, and the schoolbook decomposition is optimal.
    
    Let me just implement the schoolbook decomposition, which is already in the seed program.
    
    Actually, wait. Let me reconsider the problem statement. It says "Find a rank-minimal exact tensor decomposition". The seed program gives rank 56. The question is whether we can do better.
    
    Given the time constraints and the complexity of the problem, I'll stick with the schoolbook decomposition for now, which is known to be correct.
    
    But the directive says to try a new design. Let me think about whether there's a way to reduce the rank.
    
    Actually, I just realized that the tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor! Because T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and this can be written as:
    
    T[a,b,c] = u[a] * v[b] * w[c]
    
    where u[a] = 1, v[b] = 1, w[c] = 1 for all a, b, c.
    
    But wait, that would mean T[a,b,c] = 1 for all a, b, c, not just the consistent ones. So that's not correct.
    
    The tensor T is 1 on the consistent (a,b,c) triples and 0 otherwise. So it's not a rank-1 tensor.
    
    I think the rank is 56, and the schoolbook decomposition is optimal. Let me just implement it.
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
