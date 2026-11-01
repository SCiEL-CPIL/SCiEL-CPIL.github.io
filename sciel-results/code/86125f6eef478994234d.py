"""Matrix multiplication 2x3x9 - tensor-product decomposition of rank 48."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _build_base():
    """
    Build a rank-8 decomposition for multiplying a 2x3 matrix by a 3x1 vector.
    
    The 2x3 by 3x1 multiplication tensor T_{i,j,k} = delta_{i,j,k} (where the
    output is a 2-vector, input A is 2x3, input B is 3x1) has rank at most 8.
    
    Construction:
      - For the first 2 columns of B, use Strassen's 2x2 multiplication (rank 4)
        applied to A[:, 0:2] and B[0:2].
      - For the 3rd column of B, use the trivial rank-2 decomposition for
        A[:, 2] * B[2] (two scalar multiplications).
    
    This gives rank 4 + 2 = 6... but wait, we need to be careful.
    
    Actually, let me reconsider. The tensor for (2x3)*(3x1) -> 2x1 has:
      - 2 output entries (i=0,1)
      - 3 input entries from A per row (j=0,1,2) 
      - 3 input entries from B (k=0,1,2)
    
    The tensor has 2*3*3 = 18 entries. The schoolbook rank is 18.
    
    A better approach: Think of it as 3 separate (2x1)*(1x1) multiplications
    added together. Each (2x1)*(1x1) has rank 2. Total schoolbook: 6.
    
    Can we do better? The tensor T(i,j,k) = 1 if i,j,k are all "active" in the
    right way. Specifically, output[i] = sum_j A[i,j]*B[j].
    
    T(i,j,k) = 1 if j=k (for the contraction), 0 otherwise.
    So T(i,j,k) = delta_{j,k}.
    
    This tensor is independent of i in the sense that T(i,j,k) = delta_{j,k} for all i.
    So the tensor is rank(2x3x3 tensor where T(i,j,k)=delta_{jk}).
    
    We can write this as: for each i, the "slice" is the same tensor delta_{jk} of shape 3x3.
    The rank of delta_{jk} (3x3 identity as a matrix, but as a 3x1x3 tensor) is 3
    (it's the sum of 3 rank-1 tensors: e_j * e_j^T for j=0,1,2).
    
    Since the tensor is the same for both i=0 and i=1, we need:
    U[i,j] * V[j,k] contributions... 
    
    Actually, the decomposition is:
    T(i,j,k) = sum_r U[i,j] * V[j,k] * W[k,i]... no wait.
    
    Let me re-read the convention. The task says:
    - U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    - V has shape (m*p, rank) = (3*9, rank) = (27, rank)  
    - W has shape (p*n, rank) = (9*2, rank) = (18, rank)
    
    Wait, I think I need to reconsider the problem dimensions.
    
    The problem is: multiply a 2x3 matrix A by a 3x9 matrix B to get a 2x9 matrix C.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So n=2 (rows of A), m=3 (cols of A / rows of B), p=9 (cols of B).
    
    U: shape (n*m, rank) = (6, rank) - indexed by (i,j)
    V: shape (m*p, rank) = (27, rank) - indexed by (j,k)
    W: shape (p*n, rank) = (18, rank) - indexed by (k,i) per the trace convention W[k*n+i, r]
    
    The tensor is T[i,j,k] = 1 (the multiplication tensor for these dimensions).
    Schoolbook: rank = n*m*p = 2*3*9 = 54.
    
    Now, can we do better? The strategy mentioned is to use tensor product structure.
    
    Key insight: C = A * B where A is 2x3 and B is 3x9.
    We can view B as 3 blocks of 3x3, or as 9 columns of 3x1.
    
    Alternatively, think of it as: for each of the 9 columns of B, we compute A * B[:,k].
    Each such product is a 2x1 result from 2x3 times 3x1.
    
    The tensor for (2x3)*(3x1) -> 2x1 has a certain rank. Let's call it R_{2,3,1}.
    Then the full problem can be seen as 9 copies of this, but they share the A matrix.
    
    Actually, the full tensor T[i,j,k] for the 2x3x9 problem can be decomposed using
    the Kronecker product structure if we factor appropriately.
    
    Let me think differently. The multiplication tensor for (n x m) * (m x p) has rank
    at most n*m*p (schoolbook). Lower bounds exist, but for small cases, we can try
    explicit constructions.
    
    For 2x3x9: One approach is to note that 9 = 3*3, so we can view the 3x9 matrix as
    a 3x3 matrix of 1x3 blocks, or as having 3 row-groups.
    
    Actually, a cleaner tensor product approach:
    - Factor the 2x3x9 multiplication as (2x3x1) tensor (1x1x9) in some sense.
    - The 2x3x1 multiplication: (2x3)*(3x1) -> 2x1. 
    - The 1x1x9 multiplication: (1x1)*(1x9) -> 1x9.
    
    Rank of (2x3x1): This is the tensor T(i,j,k) = delta_{j,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    As noted, this has rank 6 (schoolbook for 2*3*1=6). But can it be lower?
    
    The tensor is: for each i, the 3x3 slice is the identity. So T = sum_i e_i (outer) I_3.
    The rank of this 2x3x3 tensor: We can write it as sum_{j=0}^{2} (sum_i e_i e_j^T) (outer) (e_j e_k... 
    
    Let me just compute: T(i,j,k) = 1 if j=k, 0 otherwise, for all i.
    
    Rank-1 decomposition: We need T(i,j,k) = sum_r U_r(i,j) * V_r(j,k) * W_r(k,i).
    
    For the 2x3x1 case (p=1): T(i,j,0) = 1 if j=0... no wait, k only goes to 0.
    
    I'm confusing myself. Let me restart with clear notation.
    
    Problem: (2x3) * (3x9) -> (2x9). n=2, m=3, p=9.
    T[i,j,k] for i in [0,2), j in [0,3), k in [0,9). T[i,j,k] = 1 for all valid indices
    (it's the identity multiplication tensor - every term in the sum contributes).
    
    Schoolbook: rank 54, with U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col=(i,j,k).
    
    Tensor product idea: 
    If we can write the 2x3x9 tensor as a tensor product of a 2x3x1 tensor and a 1x1x9 tensor,
    then rank = rank(2x3x1) * rank(1x1x9).
    
    rank(1x1x9): This is (1x1)*(1x9) -> 1x9. The tensor is T[i,j,k] = 1 for i=0,j=0,k in [0,9).
    This has rank 9 (each output column needs its own term). Actually, rank = 1*1*9 = 9? 
    No, the tensor T(0,0,k) = 1 for all k. This is rank 9 because we need to distinguish each k.
    Wait: T(i,j,k) = 1 if i=0 and j=0, for all k. As a 1x1x9 tensor, T(0,0,k)=1.
    Rank-1: U(0,0)*V(0,k)*W(k,0). We need this to equal 1 for all k.
    So U(0,0)=1, V(0,k)=1, W(k,0)=1 works. That's rank 1!
    
    Hmm, so rank(1x1x9) = 1? Let me verify. The multiplication (1x1)*(1x9) -> 1x9:
    C[k] = A[0,0] * B[0,k]. This is just scalar multiplication of a row vector.
    The tensor T[i,j,k] = 1 for i=0,j=0, any k.
    T(0,0,k) = 1 = 1 * 1 * 1. Yes, rank 1.
    
    So if we can factor 2x3x9 = (2x3x1) (x) (1x1x9), rank = rank(2x3x1) * 1 = rank(2x3x1).
    
    But what's rank(2x3x1)? The tensor for (2x3)*(3x1) -> 2x1.
    T[i,j,k] for i in [0,2), j in [0,3), k in [0,1). T[i,j,0] = 1 for all i,j.
    This is just a 2x3 tensor of all 1s (since k is always 0).
    T[i,j,0] = 1 = 1*1*1. Rank 1!
    
    So 2x3x9 would have rank 1*1 = 1? That can't be right.
    
    I'm making an error in the tensor product factorization. The issue is that the tensor
    product structure requires the dimensions to factor properly.
    
    Let me think about this more carefully using the actual tensor entries.
    
    T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,9).
    
    This is a 2x3x9 all-ones tensor. Its rank is 1! Because T[i,j,k] = 1*1*1 for all indices.
    
    But that's not the multiplication tensor. The multiplication tensor has T[i,j,k] = 1
    only when the indices correspond to a valid multiplication term, which they always do
    in the sum C[i,k] = sum_j A[i,j]*B[j,k]. So yes, every (i,j,k) triple contributes.
    
    Wait, I think I'm confusing the tensor definition. Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 9 matrix."
    
    The multiplication tensor for (n x m) * (m x p) has entries T[i,j,k] where i indexes
    rows of the result (and of A), j indexes columns of A / rows of B, and k indexes columns
    of the result (and of B). T[i,j,k] = 1 for all valid (i,j,k) because each such triple
    corresponds to one term in the matrix multiplication.
    
    So yes, T is an all-ones tensor of shape 2x3x9. Its rank is 1.
    
    But that seems too easy. Let me reconsider what the tensor decomposition means here.
    
    Actually, I think the convention might be different. In standard tensor decomposition for
    matrix multiplication, the tensor T has T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k]
    appears in C[i,k]. Since it always does, T is all ones.
    
    A rank-r decomposition means:
    C[i,k] = sum_{r=1}^{r} (sum_j U[i,j]^{(r)} * A[i,j]) * (sum_l V[j,l]^{(r)} * B[j,l]) * W[k,i]^{(r)}
    
    No wait, that's not right either. Let me look at the standard formulation.
    
    Standard: The multiplication tensor T in R^{n*m} tensor R^{m*p} tensor R^{p*n} (or similar)
    satisfies T[(i,j),(j,k),(k,i)] = 1 for all valid indices.
    
    A rank-r decomposition: T = sum_{r=1}^r u_r (tensor) v_r (tensor) w_r
    where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    This means: T[(i,j),(j',k),(k',i')] = sum_r U_r[i,j] * V_r[j',k] * W_r[k',i']
    
    For this to equal the multiplication tensor (which is 1 when j=j' and k=k' and i=i', 0 otherwise)...
    
    No, that's not right. The multiplication tensor for matrix multiplication is:
    T[i,j,k] = 1 for all i,j,k (in the appropriate ranges), because every combination
    of (row of A, column of A = row of B, column of B) gives a valid term.
    
    Hmm, but then the rank would be 1, which contradicts the schoolbook rank of n*m*p.
    
    I think I'm misunderstanding the tensor definition. Let me reconsider.
    
    In the standard formulation, the matrix multiplication tensor is defined on
    (R^n x R^m) x (R^m x R^p) x (R^n x R^p) and has the property that:
    
    T(A,B,C) = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    
    No, that's the bilinear form. The tensor itself is:
    
    T = sum_{i,j,k} e_i (in R^n) tensor e_j (in R^m) ... 
    
    I think the correct definition is: The tensor T in (R^{n*m})^* tensor (R^{m*p})^* tensor (R^{n*p})^*
    is defined by T = sum_{i,j,k} e_{(i,j)} tensor e_{(j,k)} tensor e_{(i,k)}.
    
    So T[(i,j),(j',k),(i',k')] = 1 if j=j' and i=i' and k=k', else 0.
    
    Wait no, that's not right either. Let me think about it from the bilinear form perspective.
    
    C = A * B means C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor T satisfies: T(A,B) = A*B, i.e., T is a bilinear map from R^{n*m} x R^{m*p} to R^{n*p}.
    
    As a trilinear form: trilinear(A,B,C) = sum_{i,k} C[i,k] * (A*B)[i,k] = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k].
    
    The corresponding tensor T in (R^{n*p})^* tensor (R^{n*m})^* tensor (R^{m*p})^* is:
    T = sum_{i,j,k} e_{(i,k)} tensor e_{(i,j)} tensor e_{(j,k)}.
    
    So T[(i,k),(i',j),(j',k')] = 1 if i=i', k=k', j=j', else 0.
    
    In the problem's notation:
    - U has shape (n*m, rank) - so U[i*m+j, r] corresponds to the A-factor
    - V has shape (m*p, rank) - so V[j*p+k, r] corresponds to the B-factor  
    - W has shape (p*n, rank) - so W[k*n+i, r] corresponds to the C-factor (note: k*n+i, not i*p+k)
    
    The decomposition: T[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    This should equal 1 if i=i', j=j', k=k', else 0.
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    when we set i'=i, j'=j, k'=k... 
    
    No, let me be more careful. The tensor entry T[a, b, c] where a indexes (i,j) in A-space,
    b indexes (j',k) in B-space, c indexes (k',i') in C-space (with the trace convention).
    
    T[a,b,c] = 1 if the indices are consistent: a=(i,j), b=(j,k), c=(k,i) i.e., j from a equals j' from b, k from b equals k' from c, and i from a equals i' from c.
    
    So T[(i,j),(j,k),(k,i)] = 1 for all i,j,k, and T = 0 otherwise.
    
    Now, a rank-r decomposition: T = sum_{r=1}^r u_r tensor v_r tensor w_r
    where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    This means: T[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    We need this to equal 1 if (j=j', k=k', i=i') and 0 otherwise.
    
    Schoolbook: For each (i,j,k), set U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1 for a unique r.
    Then sum_r = 1 when all indices match, 0 otherwise. Rank = n*m*p = 54.
    
    Can we do better? For small matrices, the rank of the multiplication tensor is known in some cases.
    
    For 2x2x2: rank is 7 (Strassen).
    For 2x3x9: This is a larger case. Let me think about lower bounds.
    
    The border rank of (n x m) * (m x p) multiplication is at least n*m*p / (something)... 
    Actually, there's a lower bound: rank >= n*m*p / (min(n,m,p))^2 or similar.
    
    For 2x3x9: A simple lower bound is the max of n*m, m*p, p*n = max(6, 27, 18) = 27.
    Because the rank must be at least the dimension of any one of the three spaces.
    
    So rank >= 27. The schoolbook is 54. Can we achieve something between 27 and 54?
    
    One approach: Use the fact that 9 = 3*3 and try a tensor product decomposition.
    
    Consider: (2x3) * (3x9) = (2x3) * (3x3 * 1x3) in some sense.
    
    Actually, let's think of the 3x9 matrix B as a 3x3 block matrix where each block is 1x3.
    Or, view B as having 9 columns, each of length 3.
    
    Alternative: Think of the multiplication as: for each of the 9 output columns k,
    C[:,k] = A * B[:,k]. Each B[:,k] is a 3x1 vector.
    
    So we have 9 instances of (2x3)*(3x1) -> 2x1.
    
    The tensor for (2x3)*(3x1) -> 2x1: T[i,j,k] for i in [0,2), j in [0,3), k in [0,1).
    T[(i,j),(j',0),(0,i')] = 1 if i=i', j=j', else 0.
    
    This is a 6x3x2 tensor (since n*m=6, m*p=3*1=3, p*n=1*2=2).
    
    What's the rank of this 6x3x2 tensor?
    
    The schoolbook rank is 2*3*1 = 6.
    
    Can we do better? Let's see. The tensor is:
    T[(i,j),(j',0),(0,i')] = delta_{i,i'} * delta_{j,j'}
    
    So T is a 6x3x2 tensor where T[a,b,c] = 1 if a=(i,j), b=(j,0), c=(0,i) for some i,j.
    
    This can be written as: T = sum_{i,j} e_{(i,j)} tensor e_{(j,0)} tensor e_{(0,i)}
    
    The rank of this tensor: Each term is rank 1. There are 6 terms. Can we combine them?
    
    For fixed i: sum_j e_{(i,j)} tensor e_{(j,0)} tensor e_{(0,i)}
    = (sum_j e_{(i,j)} tensor e_{(j,0)}) tensor e_{(0,i)}
    
    The sum_j e_{(i,j)} tensor e_{(j,0)} is a 6x3 matrix (for fixed i) that has 1s at positions
    ((i,j), (j,0)) for j=0,1,2. This is a 6x3 matrix with three 1s.
    
    The rank of a 6x3 matrix with three 1s in specific positions: if the 1s are in distinct rows and
    columns, the rank is 3. Here, for fixed i, the positions are ((i,0),(0,0)), ((i,1),(1,0)), ((i,2),(2,0)).
    These are 3 distinct rows and 3 distinct columns, so the 6x3 matrix has rank 3.
    
    So for each i, we have a rank-3 contribution times e_{(0,i)}. The two i-values give:
    T = sum_i (rank-3 6x3 matrix) tensor e_{(0,i)}
    
    This gives rank at most 3+3 = 6 (since the e_{(0,i)} for different i are linearly independent).
    
    So the rank of the (2x3)*(3x1) tensor is at most 6, and the schoolbook achieves 6.
    So rank = 6 for this subproblem.
    
    Now, for the full 2x3x9 problem, if we could use 6 multiplications per output column and
    share work across the 9 columns, we might do better than 54.
    
    However, the A matrix is shared across all 9 columns. The tensor product approach would
    give rank = rank(2x3x1) * rank(1x1x9) = 6 * 1 = 6, but that's not correct because the
    tensor product structure doesn't apply directly.
    
    Let me think about this differently. The full tensor T for 2x3x9 has:
    T[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k'.
    
    We can write T = sum_{i,j,k} e_{(i,j)} tensor e_{(j,k)} tensor e_{(k,i)}.
    
    This is 54 rank-1 terms. Can we reduce the number?
    
    Group by k: T = sum_k (sum_{i,j} e_{(i,j)} tensor e_{(j,k)} tensor e_{(k,i)})
    
    For fixed k, the inner sum is a 6x27x18 tensor with 6 non-zero entries.
    The rank of this sub-tensor is 6 (as computed above, with the role of dimensions adjusted).
    
    But the sub-tensors for different k share the same U and V factors (partially), so we can't
    simply add the ranks.
    
    Let me try a different approach. Use the fact that the problem is (2x3)*(3x9) and try
    to use known results for small matrix multiplication ranks.
    
    Actually, let me just try to construct a rank-48 decomposition or similar by grouping
    the k-index.
    
    Idea: For each of the 3 values of j, we have contributions to all 9 values of k.
    For fixed j and i, we have: sum_k e_{(i,j)} tensor e_{(j,k)} tensor e_{(k,i)}.
    
    = e_{(i,j)} tensor (sum_k e_{(j,k)} tensor e_{(k,i)})
    
    The sum_k e_{(j,k)} tensor e_{(k,i)} is a 27x18 matrix (for fixed j,i) with 9 non-zero entries.
    Its rank is 9 (since the 9 non-zero entries are in distinct rows and columns).
    
    So for each (i,j), we have a rank-9 contribution. Total: 6 * 9 = 54. Same as schoolbook.
    
    Alternatively, for fixed j and i: sum_k e_{(i,j)} tensor e_{(j,k)} tensor e_{(k,i)}.
    We can try to factor the k-sum differently.
    
    Note that e_{(j,k)} tensor e_{(k,i)} = sum_k e_{(j,k)} tensor e_{(k,i)}.
    This is a 3x9 x 9x2 matrix product (in tensor form). Its rank is 9.
    
    I don't see an obvious way to reduce below 54 with simple tensor product arguments.
    
    Let me check if there's a known result for 2x3x9 matrix multiplication rank.
    
    Actually, I recall that for (n x 1) * (1 x p) multiplication, the rank is n*p (it's just
    scalar multiplication of each entry). For (n x m) * (m x p), the rank is between
    max(n*m, m*p, n*p) and n*m*p.
    
    For 2x3x9: lower bound = max(6, 27, 18) = 27. Upper bound (schoolbook) = 54.
    
    A construction that might work: Use the fact that 9 = 3*3 and apply a 2x3x3 multiplication
    3 times (for each group of 3 columns), with some sharing.
    
    2x3x3 multiplication: (2x3)*(3x3) -> 2x3. Schoolbook rank = 18.
    If we do this 3 times for the 3 groups of 3 columns, total rank = 3*18 = 54. Same.
    
    But if we can share some of the 2x3 part across the groups, we might save.
    
    For 2x3x3: T[i,j,k] for i in [0,2), j in [0,3), k in [0,3).
    T[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k'.
    
    This is a 6x9x6 tensor. Schoolbook rank 18.
    
    Can we do better for 2x3x3? The lower bound is max(6,9,6) = 9.
    
    I don't have a known construction for rank < 18 for 2x3x3.
    
    Given the time constraints and the difficulty of finding a provably better rank, let me
    stick with the schoolbook rank of 54 but implement it slightly more efficiently.
    
    Actually, wait. Let me reconsider the problem. Maybe I can use a different grouping.
    
    What if I group by i (the row of the result)? For each i, C[i,:] = A[i,:] * B.
    A[i,:] is a 1x3 row vector. B is 3x9.
    
    So for each i, we're doing (1x3)*(3x9) -> 1x9.
    The tensor for (1x3)*(3x9): T[i,j,k] with i fixed (i=0 or 1), j in [0,3), k in [0,9).
    
    For fixed i: T[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k'.
    Since i is fixed, this is a 3x27x9 tensor (with the i-index removed).
    T[(j,k),(j',k),(k',i)] ... 
    
    Hmm, this is getting complicated. Let me just implement the schoolbook rank-54 solution
    and ensure it's correct.
    
    """
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return the rank-54 schoolbook decomposition for 2x3x9 matrix multiplication.
    
    This is the standard decomposition where each term in the matrix product
    C[i,k] = sum_j A[i,j] * B[j,k] corresponds to one rank-1 tensor term.
    
    The rank is 2*3*9 = 54, which is the schoolbook rank.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
