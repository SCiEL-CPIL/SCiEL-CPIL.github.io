"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    
    The task asks for U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    such that for all i, j, k:
      T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
      C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T[i, j, k] = 1 if the indices correspond to a valid multiplication term, else 0.
    Actually, the multiplication tensor has T[i, j, k] = 1 for all i, j, k because each term
    A[i,j]*B[j,k] contributes to C[i,k]. Wait, let me re-read the problem.
    
    The matrix multiplication tensor for n x m times m x p has entries:
      T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    This is because C[i,k] = sum_j A[i,j] * B[j,k], so each triple (i,j,k) contributes.
    
    The rank of this tensor is known to be at most n*m*p (schoolbook), but can be smaller.
    For small cases, the minimal rank is often n*m*p, but sometimes less.
    
    For 3x4x5, the schoolbook rank is 60. The question is whether we can do better.
    
    A known result: the rank of the matrix multiplication tensor <n,m,p> is at least 
    max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    
    Let's try to find a decomposition. One approach is to use the fact that:
    C = A * B can be computed by expressing the product in terms of rank-1 updates.
    
    Actually, let me think about this differently. The tensor T has T[i,j,k] = 1 for all i,j,k.
    We want to write T as a sum of rank-1 tensors:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    A rank-1 tensor is of the form u_i * v_j * w_k where u is a vector of length n*m,
    v is a vector of length m*p, w is a vector of length p*n.
    
    Wait, the indexing is: U has rows indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs.
    So a rank-1 term contributes U[i,j] * V[j,k] * W[k,i] to T[i,j,k].
    
    For the multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    
    So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    This is equivalent to finding matrices such that the "cyclic" product sums to 1.
    
    One simple approach: use the identity matrix trick. If we set:
      U[i,j,c] = delta_{i,c1} * delta_{j,c2} for some decomposition of the index space.
    
    Actually, the schoolbook decomposition uses one term per (i,j,k) triple:
      For each (i,j,k), set U[i,j] = 1, V[j,k] = 1, W[k,i] = 1 for that column.
    
    Can we share columns? A column c defines vectors u_c in R^(n*m), v_c in R^(m*p), w_c in R^(p*n).
    The contribution to T[i,j,k] is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i].
    
    We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    Let's consider a decomposition where we factor through the middle dimension.
    
    Idea: Write the product as a sum of rank-1 matrix products.
    A * B = sum_c (a_c * b_c^T) where a_c is n-vector, b_c is p-vector, and we need
    a_c[i] * b_c[k] to be "activated" by the j index.
    
    More precisely, if we have a decomposition where for each c, we have:
      U_c[i,j] = alpha_c[i] * beta_c[j]
      V_c[j,k] = gamma_c[j] * delta_c[k]  
      W_c[k,i] = epsilon_c[k] * zeta_c[i]
    
    Then the contribution is alpha_c[i] * beta_c[j] * gamma_c[j] * delta_c[k] * epsilon_c[k] * zeta_c[i]
    = (alpha_c[i] * zeta_c[i]) * (beta_c[j] * gamma_c[j]) * (delta_c[k] * epsilon_c[k])
    
    For this to sum to 1 over c for all i,j,k, we need the three factors to be separable,
    which means each factor must be constant (or the sum must factor).
    
    This suggests that a simple separable decomposition won't work easily.
    
    Let me try a different approach: use the fact that we can write the multiplication
    as a sequence of operations.
    
    Actually, for the specific case of 3x4x5, let's just try to find a lower-rank
    decomposition using linear algebra.
    
    The tensor T is a 3x4x5 tensor with all entries 1. We want to express it as a sum
    of rank-1 tensors in the "cyclic" sense described above.
    
    Let me set up the problem as a linear system. For a given rank r, we have r columns.
    The unknowns are the entries of U, V, W (each has r columns).
    
    The constraint is: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    This is a system of n*m*p = 60 equations with 3*r variables (if we think of each column
    as having n*m + m*p + p*n = 12 + 20 + 15 = 47 variables, so total 47*r variables).
    
    But the equations are cubic in the variables (product of three terms from different matrices),
    so it's not linear.
    
    However, we can use a trick: fix two of the matrices and solve for the third.
    Or, we can use a rank-1 update approach.
    
    Let me try a simpler approach: assume a structured decomposition.
    
    Consider the following: for the middle dimension m=4, we can write the 4x4 identity
    as a sum of rank-1 matrices. The identity I_4 has rank 4, so we need at least 4 terms
    if we decompose it as a sum of rank-1 matrices.
    
    If we write I_4 = sum_{c=1}^4 u_c * v_c^T where u_c, v_c are 4-vectors, then:
    A * B = A * I_4 * B = sum_c (A * u_c) * (v_c^T * B)
    
    Here, A * u_c is a 3-vector and v_c^T * B is a 5-vector. So each term is an outer product
    of a 3-vector and a 5-vector, which is a rank-1 3x5 matrix.
    
    So C = sum_c (A u_c) (v_c^T B)^T
    
    Now, (A u_c)[i] = sum_j A[i,j] u_c[j]
    (v_c^T B)[k] = sum_j v_c[j] B[j,k]
    
    So C[i,k] = sum_c (sum_j A[i,j] u_c[j]) * (sum_j v_c[j] B[j,k])
    
    We need this to equal sum_j A[i,j] B[j,k].
    
    This works if sum_c u_c[j] * v_c[j'] = delta_{j,j'} for all j,j', i.e., the sum of outer
    products u_c v_c^T equals the identity.
    
    So if we take u_c = e_c and v_c = e_c (standard basis vectors), then sum_c u_c v_c^T = I,
    and we get the schoolbook decomposition with rank 4. But wait, that gives rank 4, not 60?
    
    Let me check: with u_c = e_c, v_c = e_c:
    C[i,k] = sum_c (A[i,c]) * (B[c,k]) = sum_c A[i,c] B[c,k]
    
    Yes! This is exactly the matrix product, and it uses only m=4 terms!
    
    But wait, the tensor decomposition requires T[i,j,k] to be represented, not just C[i,k].
    Let me re-read the problem statement.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix."
    
    The tensor is the multiplication tensor, which encodes the bilinear map (A, B) -> C.
    The rank of this tensor is the minimum number of rank-1 tensors needed to represent it.
    
    A rank-1 tensor in this context corresponds to a rank-1 bilinear form, which is of the form
    (a^T A b) for some vectors a, b. But in the matrix multiplication case, the bilinear form
    is sum_j A[i,j] B[j,k], which involves a sum over j.
    
    The tensor T[i,j,k] = 1 for all i,j,k represents the fact that each A[i,j] and B[j,k]
    contributes to C[i,k]. The rank of this tensor is the minimum number of terms of the form
    f(i,j) * g(j,k) * h(k,i) such that sum_c f_c(i,j) * g_c(j,k) * h_c(k,i) = 1 for all i,j,k.
    
    With the identity decomposition above, let's see what f, g, h would be:
    f_c(i,j) = A[i,j] * u_c[j] = A[i,j] * delta_{j,c}
    g_c(j,k) = v_c[j] * B[j,k] = delta_{j,c} * B[j,k]
    h_c(k,i) = 1
    
    But this depends on A and B, so it's not a valid tensor decomposition. The tensor decomposition
    must be independent of the input matrices.
    
    So the correct interpretation is: the multiplication tensor T has entries T[i,j,k] = 1, and
    we want to write T as a sum of rank-1 tensors in the format described.
    
    Let me reconsider. The rank of the matrix multiplication tensor <n,m,p> is a well-studied
    problem. For small values, the rank is known or bounded.
    
    For <3,4,5>, the schoolbook rank is 60. The lower bound is max(nm, mp, pn) = max(12,20,15) = 20.
    
    I don't know the exact rank, but let's try to find a decomposition with rank less than 60.
    
    One approach: use the fact that the tensor is a "simple" tensor (all entries 1) and try to
    find a low-rank decomposition using numerical methods.
    
    Let me try to set up a least-squares problem. For a given rank r, we want to find U, V, W
    such that the reconstruction error is zero (or very small).
    
    Since the problem is cubic, we can use an alternating least squares approach:
    1. Initialize U, V, W randomly.
    2. Fix U and V, solve for W (linear least squares).
    3. Fix U and W, solve for V.
    4. Fix V and W, solve for U.
    5. Repeat until convergence.
    
    Let me implement this.
    
    The constraint is: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    Let's define:
    - For fixed U and V, the left side is linear in W. For each (k,i), we have:
      sum_c W[k*n+i, c] * (sum over j of U[i*m+j, c] * V[j*p+k, c]) = 1
      Wait, that's not quite right. Let me re-index.
    
    For a fixed (i,j,k), the equation is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    If we fix U and V, then for each (k,i), we have multiple equations (one for each j):
      sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for each j
    
    So for a fixed (k,i), we have m equations in p*n... no, in r unknowns (the entries W[k*n+i, c] for c=0..r-1).
    
    Let M_{(k,i), (j,c)} = U[i*m+j, c] * V[j*p+k, c]
    Then we need: sum_c M_{(k,i), (j,c)} * W[k*n+i, c] = 1 for each j.
    
    This is a linear system: for each (k,i), we have a matrix of size m x r (rows indexed by j,
    columns by c), and we want to find a vector w of length r such that M * w = 1 (vector of ones).
    
    We can solve this in the least-squares sense for each (k,i) independently.
    
    Similarly for the other variables.
    
    Let me implement this alternating least squares approach.
    
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook decomposition as initial guess
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by finding a lower-rank approximation
    # Use ALS with decreasing rank
    
    best_result = (rank_init, U, V, W)
    
    # Try various ranks from low to high
    min_possible = max(n * m, m * p, p * n)
    
    for target_rank in range(min_possible, rank_init + 1):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random or structured guess
        np.random.seed(target_rank)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # ALS iterations
        for it in range(50):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    # Build the m x target_rank matrix
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # Solve M * w = ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n * p, target_rank))  # Wait, this isn't right
                    # Let me re-derive. For fixed (j,k), the equations are over i:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each i
                    # So for fixed (j,k), we have n equations in target_rank unknowns.
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        for c in range(target_rank):
                            M[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        for c in range(target_rank):
                            M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass
        
        # Check if we got an exact solution
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(target_rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += abs(val - 1)
        
        if error < 1e-6:
            best_result = (target_rank, U.copy(), V.copy(), W.copy())
            break  # Found a valid decomposition, stop
    
    rank, U, V, W = best_result
    
    # Convert to lists
    U_list = [[U[i, j] for j in range(rank)] for i in range(n * m)]
    V_list = [[V[i, j] for j in range(rank)] for i in range(m * p)]
    W_list = [[W[i, j] for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
