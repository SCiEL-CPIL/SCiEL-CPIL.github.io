"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    The key insight: a 4x9 matrix can be viewed as nine independent 4x1 columns.
    Multiplying a 4x4 matrix A by a 4x9 matrix B is equivalent to computing
    A @ B[:, j] for j = 0, ..., 8. Each of these is a 4x4 @ 4x1 multiplication.
    
    For a 4x4 @ 4x1 multiplication, we can use the Strassen-based rank-18 
    decomposition for 4x4x4 matrix multiplication restricted to one column,
    or more simply, we can use the standard rank-16 schoolbook for 4x4 @ 4x1.
    
    Actually, let's think more carefully. The tensor for 4x4 @ 4x9 is:
    T_{(i,j),k} = sum_l A_{i,l} B_{l,j}
    
    This is the same as doing 9 independent 4x4 @ 4x1 multiplications.
    The rank of the 4x4 @ 4x1 tensor is 16 (schoolbook), and since the columns
    are independent, the total rank is 9 * 16 = 144. That's the same as schoolbook.
    
    But wait - we can do better. The 4x4 @ 4x9 multiplication can be seen as
    multiplying a 4x4 matrix by a 4x9 matrix. We can use a block decomposition.
    
    Let's use the fact that we can compute A @ B where B is 4x9 by using
    a rank-18 decomposition for 4x4x4, applied to three blocks of B:
    B = [B1, B2, B3] where B1 is 4x3, B2 is 4x3, B3 is 4x3.
    
    But that doesn't directly help. Let me think differently.
    
    Actually, the best approach: use the known rank-18 decomposition for 4x4x4.
    A 4x9 matrix can be padded to 4x12 (three 4x4 blocks). Then A @ B_padded
    gives a 4x12 result, and we take the first 9 columns.
    
    With rank-18 for 4x4x4, multiplying 4x4 by 4x12 requires 3 independent
    4x4x4 multiplications, so rank = 3 * 18 = 54.
    
    Let me implement the rank-18 decomposition for 4x4x4.
    
    The rank-18 decomposition of 4x4x4 matrix multiplication can be derived
    from the Strassen algorithm. For 2x2, Strassen uses 7 multiplications.
    For 4x4, we can use a 2x2 block structure where each block is 2x2,
    and apply Strassen at the block level, with each block multiplication
    also using Strassen. That gives 7*7 = 49, which is not 18.
    
    The rank-18 decomposition is a known result (Bini, Capovani, Lotti, Romani 1979).
    Let me look up the specific structure.
    
    Actually, the optimal rank for 4x4x4 is known to be 18 (for the exact decomposition).
    Let me construct it.
    
    The 4x4x4 tensor of rank 18:
    We need 18 rank-1 tensors (u_r, v_r, w_r) such that:
    T_{i,j,k} = sum_r u_r[i] * v_r[j] * w_r[k]
    where T_{i,j,k} = delta_{i+k, j} (with appropriate indexing for matrix mult)
    
    For matrix multiplication C = A * B where A is 4x4, B is 4x4, C is 4x4:
    C[i,j] = sum_k A[i,k] * B[k,j]
    So T_{(i,j),k,l} = A[i,k]*B[k,l] contributes to C[i,l].
    The tensor is: sum_k A[i,k] B[k,l] = C[i,l]
    In tensor form: T_{i,l,k} = delta_{i+l?} no...
    
    Let me use the standard formulation:
    The matrix multiplication tensor M_{n} has entries M_{(i,j),(k,l)} = delta_{j,k}
    for i,j,k,l in {0,...,n-1}.
    
    For n=4, we need rank 18.
    
    Let me use a known rank-18 decomposition. One construction:
    
    Using the 3x3 Strassen-like decomposition for 2x2 blocks:
    Actually, let me just use the fact that rank(4,4,4) = 18 and find
    the explicit decomposition.
    
    Here's a known rank-18 decomposition for 4x4x4:
    
    The 18 terms can be organized as follows. Consider the 4x4 matrices
    partitioned into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Using Strassen at the block level (7 products of 2x2 matrices),
    and using Strassen at the scalar level for each 2x2 product (7 multiplications),
    we get 49 terms. But we want 18.
    
    The rank-18 decomposition is not simply a composition of two Strassen steps.
    It's a more efficient decomposition.
    
    Let me try a different approach: I'll use the known result that
    the 4x4x4 multiplication tensor has rank 18, and I'll construct it
    using a known explicit formula.
    
    Actually, for the purpose of this problem, let me just use the schoolbook
    decomposition for the 4x4 part (rank 16) and see if I can get a total
    rank better than 144.
    
    For 4x4 @ 4x9:
    If I use schoolbook for each of the 9 columns: 9 * 16 = 144.
    If I pad to 4x12 and use 3 * 16 = 48 for the 4x4x4 parts, total = 48.
    Wait, that's not right either. Let me reconsider.
    
    A is 4x4, B is 4x9. C = A @ B is 4x9.
    C[i,j] = sum_k A[i,k] * B[k,j] for i=0..3, j=0..8, k=0..3.
    
    This is a tensor of size (4*9) x (4*4) x (4*9) = 36 x 16 x 36.
    The rank of this tensor is what we want to minimize.
    
    Using schoolbook: for each (i,j,k) triple, one term. Total = 4*9*4 = 144.
    
    Can we do better? Yes! We can share computations across different j values.
    For a fixed (i,k), the value A[i,k] is multiplied by B[k,j] for all j.
    So we can define P_{i,k} = A[i,k] (no change), but we can compute
    S_{i,j} = sum_k A[i,k] B[k,j] by first computing T_{k,j} = ... 
    
    Actually, the key insight is: for each k, we compute A[:,k] * B[k,:]
    which is an outer product of a 4x1 and 1x9 vector, giving a 4x9 matrix.
    We sum over k=0..3. Each outer product has rank 1 (as a 4x9 matrix),
    so the sum of 4 rank-1 matrices has rank at most 4.
    
    But in terms of the tensor decomposition, each term A[i,k]*B[k,j]
    for fixed (i,k,j) is a rank-1 term in the tensor.
    
    Let me think about this differently. The tensor is:
    T_{(i,j),k} = sum_k A[i,k] B[k,j]
    
    This is equivalent to: for each k, we have a contribution
    A[:,k] (outer) B[k,:] to the result C.
    
    In tensor decomposition terms, for each k, we have:
    U_{(i,k),k} = A[i,k] contribution... 
    
    Let me use the standard tensor decomposition framework.
    For matrix multiplication C = A @ B, the tensor is:
    M = sum_{i,j,k} e_i e_j^T \otimes e_k e_k^T \otimes e_k e_j^T
    
    No, let me be more careful. The matrix multiplication tensor for
    n x n matrices is:
    M_n = sum_{i,j,k=1}^n e_i \otimes e_k \otimes e_k \otimes e_j
    where e_i is the standard basis vector.
    
    In the format used here:
    U has shape (n*m, rank) = (4*4, rank) = (16, rank)
    V has shape (m*p, rank) = (4*9, rank) = (36, rank)
    W has shape (p*n, rank) = (9*4, rank) = (36, rank)
    
    The decomposition satisfies:
    for all (i,j) in U's rows, (k,l) in V's rows, (m,n) in W's rows:
    sum_r U[(i,k),r] * V[(k,l),r] * W[(l,j),r] = delta_{i,j?}
    
    Wait, the indexing is:
    U[i*m + j, r] where i is row of A, j is column of A (= row of B)
    V[j*p + k, r] where j is row of B, k is column of B
    W[k*n + i, r] where k is column of B, i is row of C (= row of A)
    
    And the condition is:
    C[i, k] = sum_j A[i,j] B[j,k]
    So: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j?}
    
    Actually, the condition for the tensor decomposition of matrix multiplication
    A (n x m) @ B (m x p) = C (n x p) is:
    
    C[i,p_idx] = sum_j A[i,j] B[j,p_idx]
    
    The tensor entries are T_{(i,j),(j',p_idx)} = delta_{j,j'}
    where (i,j) indexes A, (j',p_idx) indexes B, and the output is (i,p_idx).
    
    In the W indexing: W[k*n + i, r] where k = p_idx, i = row of A/C.
    
    So the condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j, j} = 1
    when the indices match properly.
    
    More precisely:
    For the decomposition to be valid:
    sum_r U[(i,j),r] V[(j,k),r] W[(k,i),r] = 1 for all i,j,k? No...
    
    Let me re-read the problem. The tensor for multiplying an n x m matrix
    by an m x p matrix is:
    T_{(i,k), (j), (l)} where i is row of A, j is column of A / row of B,
    k is column of B, l is row of C.
    
    Actually, the standard formulation:
    (A @ B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor M has entries M_{(i,j),(j,k)} = delta_{j,j} (always 1 when indices match)
    and the "output" index is (i,k).
    
    In the decomposition:
    U has rows indexed by (i,j) where i in [0,n), j in [0,m)
    V has rows indexed by (j,k) where j in [0,m), k in [0,p)
    W has rows indexed by (k,i) where k in [0,p), i in [0,n)
    
    And: sum_r U[(i,j),r] V[(j,k),r] W[(k,i),r] = 1 for all valid i,j,k
    (this ensures that the product reconstructs correctly)
    
    Wait, that's not quite right either. Let me think about it as:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    If we write A_{i,j} = sum_r U[(i,j),r] * something, that's not how it works.
    
    The correct formulation: the matrix multiplication tensor T is a trilinear form:
    T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} (with appropriate signs/indices)
    
    No, the standard tensor decomposition for matrix multiplication:
    We want to compute C = A @ B.
    T = sum_{i,j,k} A_{i,j} \otimes B_{j,k} \otimes C_{k,i}^*
    where C^* is the dual.
    
    For the rank decomposition:
    T = sum_{r=1}^rank u_r \otimes v_r \otimes w_r
    
    where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    The condition is:
    For all i,j,k: sum_r U[(i,j),r] V[(j,k),r] W[(k,i),r] = 1
    
    This ensures that the bilinear form reconstructs the matrix product.
    
    OK so for schoolbook:
    For each (i,j,k), one term with U[(i,j),r]=1, V[(j,k),r]=1, W[(k,i),r]=1.
    Total rank = n*m*p = 4*4*9 = 144.
    
    Now, can we do better? The key is to find linear dependencies.
    
    For fixed i,k: sum_j A_{i,j} B_{j,k}
    The terms for different j are not independent in a way that allows us to
    reduce the rank below n*m*p in general, but for specific dimensions,
    we might.
    
    Actually, the rank of the matrix multiplication tensor M_{n,m,p} is
    known for small values. For M_{4,4,9}, I'm not sure of the exact rank.
    
    Let me try a different approach: use the fact that 4x9 = 4x(4+4+1)
    and compute three 4x4 multiplications and one 4x1 multiplication.
    
    A @ B = [A @ B1, A @ B2, A @ B3] where B1, B2 are 4x4 and B3 is 4x1.
    
    For each 4x4 multiplication, use the rank-18 decomposition.
    For the 4x1 multiplication, use schoolbook (rank 16).
    
    Total rank = 2*18 + 16 = 52.
    
    But wait, the 4x1 multiplication: C_{i,0} = sum_j A_{i,j} B_{j,0}
    This is just a vector-matrix product, and the rank is 4*4*1 = 16.
    
    So total rank = 2*18 + 16 = 52.
    
    Let me implement the rank-18 decomposition for 4x4x4.
    
    I'll use a known construction. The 4x4x4 tensor of rank 18 can be
    constructed using the following method:
    
    Partition each 4x4 matrix into four 2x2 blocks.
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Using Strassen at the block level:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 - M4 + M6 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M5
    
    Each M_r is a 2x2 matrix product, which can be computed with 7 scalar
    multiplications using Strassen. So total = 7*7 = 49.
    
    But we want rank 18, not 49. The rank-18 decomposition is a more
    efficient one that doesn't simply compose two Strassen steps.
    
    Let me look for an explicit rank-18 decomposition.
    
    After some research, the rank-18 decomposition of 4x4x4 is known but
    not trivial to write down from memory. Let me try a different approach.
    
    Since I'm not sure I can correctly implement the rank-18 decomposition
    from scratch, let me use a safer approach:
    
    1. Use schoolbook for the entire 4x4 @ 4x9: rank 144.
    2. Try to improve by using the block structure.
    
    Actually, let me try yet another approach. The 4x9 matrix B can be
    thought of as a 4x4 matrix B1 concatenated with a 4x4 matrix B2 and
    a 4x1 matrix B3.
    
    If I use a rank-r decomposition for 4x4x4, I need 2*r + 16 for the total.
    If r = 16 (schoolbook for 4x4x4), total = 2*16 + 16 = 48.
    If r = 18, total = 2*18 + 16 = 52.
    
    Wait, that's worse! Schoolbook for 4x4x4 has rank 16, so 2*16 + 16 = 48
    is better than 2*18 + 16 = 52.
    
    But can I do even better? What if I use a rank-16 decomposition for
    4x4x4 and share some terms between the two 4x4 blocks?
    
    Actually, the two 4x4 multiplications are independent, so I can't share
    terms between them. The total rank is at least 2*16 + 16 = 48.
    
    But wait, maybe I can do better by not splitting B into 4x4 + 4x4 + 4x1.
    What if I use a different block structure?
    
    Let me try: B is 4x9. I can view it as a 4x3 block times a 3x3 "identity-like"
    matrix, but that doesn't help.
    
    Alternatively, I can use the fact that the rank of M_{4,4,9} might be
    less than 48. Let me try to compute it using a simple optimization.
    
    Actually, for the purposes of this task, let me just implement the
    block decomposition with schoolbook for each 4x4 block and the 4x1 part.
    
    Total rank = 2 * 16 + 16 = 48.
    
    This is a significant improvement over 144.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll split B into three parts: B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A @ B = [A @ B1, A @ B2, A @ B3]
    # Each A @ Bx is a 4x4 or 4x1 multiplication.
    
    # For each 4x4 multiplication, use schoolbook (rank 16)
    # For the 4x1 multiplication, use schoolbook (rank 16)
    
    # Total rank = 16 + 16 + 16 = 48
    
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 48
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 48
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 48
    
    col = 0
    
    # Part 1: A @ B1 where B1 is columns 0-3 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):  # k is 0..3, column of B1
                # U[i*m + j, col] = 1 (A[i,j])
                # V[j*p + k, col] = 1 (B[j,k]) where k is 0..3
                # W[k*n + i, col] = 1 (C[i,k]) where k is 0..3
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Part 2: A @ B2 where B2 is columns 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):  # k is 4..7, column of B
                # U[i*m + j, col] = 1 (A[i,j])
                # V[j*p + k, col] = 1 (B[j,k]) where k is 4..7
                # W[k*n + i, col] = 1 (C[i,k]) where k is 4..7
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Part 3: A @ B3 where B3 is column 8 of B
    for i in range(n):
        for j in range(m):
            k = 8  # k is 8, column of B
            # U[i*m + j, col] = 1 (A[i,j])
            # V[j*p + k, col] = 1 (B[j,k]) where k is 8
            # W[k*n + i, col] = 1 (C[i,k]) where k is 8
            U[i * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
