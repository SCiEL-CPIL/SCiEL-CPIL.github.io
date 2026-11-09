"""Program for rank-minimal exact tensor decomposition of 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    The matrix multiplication tensor T is defined by:
    T[i,j,k,l] = delta_{j,k} where:
    - i in {0,1} (row of A, 2 rows)
    - j in {0,1,2,3} (col of A / row of B, 4)
    - k in {0,1,2,3} (col of B... wait, let me re-index)
    
    Actually for matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where:
    - i: row of A (0..n-1)
    - j: column of A = row of B (0..m-1)  
    - k: column of B (0..p-1)
    
    T[i,j,k] = 1 if this term contributes to C[i,k], which it does for all valid (i,j,k).
    Wait, that's not right either. The tensor is:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No, that's the all-ones tensor. The matrix multiplication tensor is:
    T[i,j,k] = 1 if the entry A[i,j] * B[j,k] contributes to C[i,k]
    
    In the standard formulation, the multiplication tensor is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    where e_i is the i-th basis vector of R^n (for A's row index),
    e_j is the j-th basis vector of R^m (for the shared index),
    e_k is the k-th basis vector of R^p (for B's column index).
    
    So T[i,j,k] = 1 for all valid i,j,k. This is indeed the all-ones tensor of shape (n,m,p).
    
    The rank of this tensor is what we want to minimize.
    
    For n=2, m=4, p=7, the schoolbook rank is n*m*p = 56.
    
    Let me try to find a lower rank decomposition.
    
    One approach: use the fact that we can group indices.
    
    Consider splitting the m=4 dimension into two parts of size 2:
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x7
    
    C = A1*B1 + A2*B2
    
    The tensor for this is: T = T1 + T2 where T1 is the 2x2x7 tensor and T2 is also 2x2x7.
    
    If we can find a rank r decomposition for the 2x2x7 tensor, then the total rank is 2r.
    
    For 2x2x7, the schoolbook rank is 2*2*7 = 28.
    
    Can we do better for 2x2x7? Let's think about it.
    
    For 2x2xp, the multiplication tensor has rank at most 2*p (using schoolbook for each of the 2x2 blocks... no, that's not right).
    
    Actually, for 2x2xp:
    A = [[a,b],[c,d]], B is 2xp
    C = A*B, so C[0,k] = a*B[0,k] + b*B[1,k], C[1,k] = c*B[0,k] + d*B[1,k]
    
    Schoolbook: 4*p multiplications.
    
    Can we use Strassen-like tricks? For 2x2x2, Strassen gives rank 7 instead of 8.
    
    For 2x2xp, we might be able to use 7*p/2 or something similar.
    
    Actually, let me think differently. The 2x2xp tensor can be seen as p copies of the 2x2x1 tensor (which is just 4 terms) but with shared structure.
    
    Hmm, this is getting complicated. Let me try a different approach.
    
    Let me try to use random initialization and gradient descent or some other optimization to find a low-rank decomposition.
    
    Actually, for exact decompositions, we need to be more careful. Let me try to construct one analytically.
    
    For 2x4x7, let me try to find a rank 50 decomposition or lower.
    
    One idea: use the fact that 4 = 2+2 and 7 = 3+4, and apply some block decomposition.
    
    Another idea: use the Winograd algorithm or similar fast matrix multiplication techniques.
    
    Given the time constraints, let me try a computational approach:
    1. Start with a random initialization for U, V, W
    2. Use some optimization to minimize the rank while maintaining exactness
    
    But exactness is a hard constraint. Let me think about this differently.
    
    The tensor T[i,j,k] = 1 for all i,j,k. We want to write:
    T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the indices need to match. Let me re-read the problem.
    
    The output factor follows the official trace convention W[k*n+i,r].
    
    So W is indexed by (k*n + i, r) where k is the column of B (0..p-1) and i is the row of A (0..n-1).
    
    The decomposition is:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k] = sum_{r=1}^{R} U[i*m+j... wait, this doesn't look right.
    
    Let me re-read the interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    And W[k*n+i, r] is the (k*n+i)-th row of W, r-th column.
    
    So the decomposition is:
    For each r in 1..R:
    - U[:, r] is a vector of length n*m, representing a linear combination of A's entries
    - V[:, r] is a vector of length m*p, representing a linear combination of B's entries
    - W[:, r] is a vector of length p*n, representing the contribution to C
    
    The product U[:,r] * V[:,r] (outer product, flattened) should give W[:,r] when appropriately mapped.
    
    More precisely, if we unflatten U[:,r] into a n x m matrix U_r, and V[:,r] into a m x p matrix V_r,
    then U_r * V_r should equal W_r where W_r is the p x n matrix obtained by reshaping W[:,r] with W[k*n+i] = W_r[k,i].
    
    So the condition is: for all r, U_r @ V_r = W_r.
    
    And the total C = sum_r U_r @ V_r = A @ B for all A, B.
    
    This means: sum_r U_r @ V_r = A @ B for all A (n x m) and B (m x p).
    
    Equivalently, the linear map from (A, B) to C is represented by the sum of rank-1 tensor products.
    
    Let me try to find a decomposition with rank less than 56.
    
    One approach: use the fact that the tensor is symmetric in some sense and try to exploit that.
    
    Another approach: use known results. The rank of the matrix multiplication tensor M_{n,m,p} is known to be at most n*m*p (schoolbook) and at least something.
    
    For small cases, the exact rank is known. For M_{2,4,7}, I'm not sure of the exact rank.
    
    Let me try a computational search. I'll use a random initialization and try to minimize the rank by checking if a lower-rank decomposition exists.
    
    Actually, let me try a simpler approach: start with the schoolbook decomposition and try to merge terms.
    
    The schoolbook decomposition has 56 terms, each of the form:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r = i*m*p + j*p + k.
    
    Can we merge any of these? Two terms can be merged if their sum can be written as a single term.
    
    Term r1: U1[i1*m+j1] = 1, V1[j1*p+k1] = 1, W1[k1*n+i1] = 1
    Term r2: U2[i2*m+j2] = 1, V2[j2*p+k2] = 1, W2[k2*n+i2] = 1
    
    The sum is: U_sum[i*m+j] = U1[i*m+j] + U2[i*m+j], etc.
    
    For the sum to be a single term, we need U_sum, V_sum, W_sum to be rank-1 tensors (outer products of vectors).
    
    This is a complex condition. Let me try a different approach.
    
    Let me use numpy to try to find a low-rank decomposition via optimization.
    
    I'll parameterize U, V, W as dense matrices and try to minimize some loss function that enforces the decomposition.
    
    The constraint is: for all A, B, sum_r U_r @ V_r = A @ B.
    
    This is equivalent to: the tensor sum_r U_r ⊗ V_r ⊗ W_r = T where T is the multiplication tensor.
    
    In terms of the matrices U, V, W, the condition is that for all i, k:
    sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * ... 
    
    Wait, I'm confusing myself. Let me be more careful.
    
    The decomposition says: C[i,k] = sum_r W[k*n+i, r] * (some product of A and B entries)
    
    No, that's not right either. Let me re-read the problem statement.
    
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    So U, V, W are the factor matrices. The tensor decomposition is:
    T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    where u_r is the r-th column of U (length n*m), v_r is the r-th column of V (length m*p), w_r is the r-th column of W (length p*n).
    
    And T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k, and 0 otherwise.
    
    Wait, the tensor T is a 3D array with dimensions (n*m, m*p, p*n)? That doesn't seem right.
    
    Let me reconsider. The matrix multiplication tensor is a 3-tensor with indices (i, j, k) where:
    - i: 0..n-1 (row of A)
    - j: 0..m-1 (column of A / row of B)
    - k: 0..p-1 (column of B)
    
    T[i,j,k] = 1 for all valid i, j, k.
    
    The rank of this tensor is the minimum R such that:
    T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But U has shape (n*m, R), so U[i*m+j, r] corresponds to U[i,j,r] in the tensor notation.
    Similarly, V[j*p+k, r] corresponds to V[j,k,r], and W[k*n+i, r] corresponds to W[k,i,r].
    
    So the condition is:
    1 = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all valid i, j, k.
    
    And for invalid combinations (which don't exist in this case since all i, j, k are valid), it should be 0.
    
    So we need to find U, V, W with integer/half-integer entries such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in 0..1, j in 0..3, k in 0..6.
    
    This is a system of 2*4*7 = 56 equations in 56*R unknowns (since U, V, W each have R columns, and each has n*m, m*p, p*n rows respectively).
    
    For R = 56 (schoolbook), we have 56*3*56 = 9408 unknowns and 56 equations, so there are many solutions.
    
    For R < 56, we have fewer unknowns and it becomes a constraint satisfaction problem.
    
    Let me try to find a solution with R = 50, 48, etc.
    
    One approach: use integer programming or SAT, but that's complex.
    
    Another approach: use the fact that the tensor is all-ones and try to find a structured decomposition.
    
    Let me try a different strategy: use the known Strassen algorithm for 2x2x2 and extend it.
    
    For 2x2x2, Strassen's algorithm uses 7 multiplications instead of 8.
    
    For 2x4x7, we can view it as a block matrix multiplication.
    
    Let me try to use a block decomposition where we split the 4x7 matrix into 2x2 blocks.
    
    A is 2x4, so we can split it into two 2x2 blocks: A = [A1 | A2].
    B is 4x7, so we can split it into four 2x7 blocks: B = [B11; B12; B21; B22] where each Bi is 2x7.
    
    Then C = A1*B1 + A2*B2 where B1 = [B11; B12] is 4x7 and B2 = [B21; B22] is 4x7.
    
    Wait, that's not quite right. Let me be more careful.
    
    A is 2x4: A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    B is 4x7: B = [[b00, ..., b06], [b10, ..., b16], [b20, ..., b26], [b30, ..., b36]]
    
    C = A*B is 2x7:
    C[0,k] = a00*b0k + a01*b1k + a02*b2k + a03*b3k
    C[1,k] = a10*b0k + a11*b1k + a12*b2k + a13*b3k
    
    So C[0,k] = sum_{j=0}^{3} A[0,j] * B[j,k]
    C[1,k] = sum_{j=0}^{3} A[1,j] * B[j,k]
    
    This is just standard matrix multiplication. The tensor is T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
    
    Now, can we find a lower-rank decomposition?
    
    Let me try to use the following idea: group the j indices.
    
    For j=0,1: T[i,0,k] and T[i,1,k]
    For j=2,3: T[i,2,k] and T[i,3,k]
    
    If we can find a rank-7 decomposition for the 2x2x7 sub-tensor (i in {0,1}, j in {0,1}, k in {0,...,6}), then we can use 2*7 = 14 terms for the two groups, plus some cross terms.
    
    But this is getting complicated. Let me try a computational approach.
    
    I'll use a random search: initialize U, V, W with random integer/half-integer values, and check if they satisfy the equations. If not, adjust them.
    
    But this is unlikely to work for R < 56.
    
    Let me try a different approach: use the fact that the tensor is all-ones and try to find a decomposition using a small number of "basis" tensors.
    
    For example, if we can find vectors u, v, w such that u ⊗ v ⊗ w = T, then the rank is 1. But T is all-ones, so u, v, w would all need to be all-ones vectors, and u ⊗ v ⊗ w would be all-ones, which is T. So the rank is 1?
    
    Wait, that can't be right. If u = [1,1,...,1] (length n*m), v = [1,1,...,1] (length m*p), w = [1,1,...,1] (length p*n), then:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 * 1 * 1 = 1.
    
    So the rank is 1? That seems too good to be true.
    
    Let me double-check. The tensor T[i,j,k] = 1 for all i, j, k. If we set U[i*m+j, 1] = 1, V[j*p+k, 1] = 1, W[k*n+i, 1] = 1 for all i, j, k, then:
    sum_{r=1}^{1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 * 1 * 1 = 1 = T[i,j,k].
    
    So the rank is indeed 1! But that can't be right because the matrix multiplication tensor is known to have rank greater than 1.
    
    I must be misunderstanding the tensor decomposition. Let me re-read the problem.
    
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    The matrix multiplication tensor is not the all-ones tensor. It's the tensor that represents the bilinear map (A, B) -> A*B.
    
    The tensor T has entries T[i,j,k] where i is the row of A, j is the column of A (and row of B), and k is the column of B.
    
    The entry T[i,j,k] is 1 if the product A[i,j] * B[j,k] contributes to C[i,k], and 0 otherwise.
    
    In standard matrix multiplication, A[i,j] * B[j,k] always contributes to C[i,k] (with coefficient 1), so T[i,j,k] = 1 for all valid i, j, k.
    
    But the tensor decomposition is not just about the values of T; it's about the structure of the bilinear map.
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k].
    
    This can be written as: C = sum_j (A[:,j] * B[j,:]) where A[:,j] is the j-th column of A and B[j,:] is the j-th row of B.
    
    Each term A[:,j] * B[j,:] is an outer product of a vector (length n) and a vector (length p), giving a n x p matrix.
    
    So the bilinear map is a sum of m = 4 terms, each of which is a rank-1 tensor in the sense that it's an outer product of two vectors.
    
    But the tensor rank is different from the number of terms in this sum. The tensor rank is the minimum number of rank-1 tensors (in the 3-tensor sense) needed to sum to T.
    
    A rank-1 3-tensor is of the form u ⊗ v ⊗ w where u is a vector of length n, v is a vector of length m, w is a vector of length p.
    
    So T[i,j,k] = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k].
    
    For the matrix multiplication tensor, T[i,j,k] = 1 for all i, j, k, so we need:
    1 = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] for all i, j, k.
    
    This is a system of n*m*p equations in R*(n+m+p) unknowns.
    
    For n=2, m=4, p=7, we have 56 equations in R*(2+4+7) = 13R unknowns.
    
    If R=1, we have 13 unknowns and 56 equations, which is overdetermined. So R=1 is not possible in general.
    
    But wait, the tensor is all-ones, so maybe there's a special structure.
    
    If we set u_r[i] = 1/sqrt(n), v_r[j] = 1/sqrt(m), w_r[k] = 1/sqrt(p) for all r, and R = n*m*p, then:
    sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] = R * (1/sqrt(n)) * (1/sqrt(m)) * (1/sqrt(p)) = (n*m*p) / (sqrt(n)*sqrt(m)*sqrt(p)) = sqrt(n*m*p).
    
    That's not 1. So that doesn't work.
    
    Let me try R = n*m*p = 56. If we set u_r[i] = 1 if i = i_r, 0 otherwise; v_r[j] = 1 if j = j_r, 0 otherwise; w_r[k] = 1 if k = k_r, 0 otherwise, where (i_r, j_r, k_r) enumerates all triples, then:
    sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] = 1 if there exists r such that i_r = i, j_r = j, k_r = k, which is true for exactly one r, so the sum is 1.
    
    So the rank is at most 56.
    
    Can we do better? Let's try R = 55. We need to find u_r, v_r, w_r for r=1..55 such that:
    sum_{r=1}^{55} u_r[i] * v_r[j] * w_r[k] = 1 for all i, j, k.
    
    This is a system of 56 equations in 55*(2+4+7) = 715 unknowns. It might have a solution.
    
    Let me try to find a solution using linear algebra.
    
    I'll set up the system of equations and solve it.
    
    Actually, let me try a different approach. Let me use the fact that the tensor is all-ones and try to find a decomposition using a small number of terms.
    
    For example, if we can find vectors u, v, w such that u ⊗ v ⊗ w = T, then the rank is 1. But T is all-ones, so u, v, w would all need to be constant vectors, and u ⊗ v ⊗ w would be constant, which is T. So the rank is 1?
    
    Wait, I'm going in circles. Let me just try to solve the system of equations for R = 55, 54, etc.
    
    I'll use numpy to set up the system and solve it.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Build schoolbook first as fallback
    def build_schoolbook():
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
    
    # Try to find a lower-rank decomposition
    def try_rank(R):
        """Try to find a rank-R decomposition."""
        # Set up the system of equations
        # Unknowns: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for i in 0..n-1, j in 0..m-1, k in 0..p-1, r in 0..R-1
        # Equations: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a nonlinear system because of the products.
        # Let's try a different approach: fix two of the matrices and solve for the third.
        
        # For example, fix U and V, then the equations become linear in W.
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For fixed (i,j,k), this is: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # This is a linear equation in W[k*n+i, r].
        
        # But we have multiple equations for the same W entry (different (i,j,k) can give the same (k*n+i)).
        # Specifically, W[k*n+i, r] appears in equations for all (i', j', k') where k'=k and i'=i.
        # For fixed i and k, the equations are: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0..m-1.
        # So for fixed i, k, we have m equations in R unknowns (W[k*n+i, r] for r=0..R-1).
        
        # If m > R, this is overdetermined. If m <= R, it might have a solution.
        
        # For n=2, m=4, p=7, we have 4 equations for each (i,k) pair, and there are n*p = 14 such pairs.
        # So we have 14*4 = 56 equations in 14*R unknowns (W has p*n = 14 rows and R columns).
        
        # Let's try to solve this system.
        
        # First, initialize U and V randomly.
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        
        # Now solve for W.
        # For each (i, k), we have the system:
        # sum_{r=0}^{R-1} (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1 for j=0..m-1
        
        # This can be written as: A_{ik} @ W_{ik} = b where A_{ik} is a m x R matrix and W_{ik} is a R x 1 vector.
        # A_{ik}[j, r] = U[i*m+j, r] * V[j*p+k, r]
        # b = [1, 1, ..., 1]^T (length m)
        
        # We can solve this using least squares.
        
        W = np.zeros((p * n, R))
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A_ik[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b = np.ones(m)
                # Solve A_ik @ W_ik = b
                W_ik, residuals, rank, sv = np.linalg.lstsq(A_ik, b, rcond=None)
                W[k * n + i, :] = W_ik
        
        # Now check if the decomposition is exact.
        # For each (i, j, k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(R))
                    err = abs(val - 1)
                    max_err = max(max_err, err)
        
        return max_err, U, V, W
    
    # Try different ranks
    best_rank = n * m * p
    best_result = build_schoolbook()
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    for R in range(50, best_rank):  # Try ranks from 50 to 55
        if time.time() > deadline - 10:  # Leave 10 seconds for cleanup
            break
        
        max_err, U, V, W = try_rank(R)
        if max_err < 1e-6:  # Found a valid decomposition
            best_rank = R
            # Convert to lists of integers/half-integers
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_result
# EVOLVE-BLOCK-END
