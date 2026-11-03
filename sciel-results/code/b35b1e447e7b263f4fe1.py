"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    The tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 2)
    - j in [0, m) = [0, 3)  
    - k in [0, p) = [0, 6)
    
    T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i, j, k] = 1 for all i, j, k (it's the full outer product structure).
    Wait, let me reconsider. The matrix multiplication tensor has entries:
    T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    So the tensor is T[i,j,k] = 1 for all valid i,j,k. This is actually a rank-1 tensor!
    No wait, that's not right either. Let me think more carefully.
    
    The matrix multiplication tensor M_{n,m,p} has entries:
    M[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    
    So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    This means the tensor is actually the all-ones tensor of shape (n, m, p).
    
    The rank of the all-ones tensor of shape (n,m,p) is... let's think.
    
    A rank-1 tensor has the form u_i * v_j * w_k. The all-ones tensor can be written as:
    1 * 1 * 1 for each entry, so it's rank 1? No, that's not how tensor rank works.
    
    Actually, the tensor T[i,j,k] = 1 for all i,j,k can be decomposed as:
    T = sum_c U[i,c] * V[j,c] * W[k,c]
    
    For this to equal 1 for all i,j,k, we need:
    sum_c U[i,c] * V[j,c] * W[k,c] = 1
    
    One way: take rank = 1, U[i,0] = 1, V[j,0] = 1, W[k,0] = 1. Then sum = 1*1*1 = 1. 
    So the rank is 1!
    
    Wait, but the problem says the schoolbook decomposition has rank n*m*p = 36. That doesn't match.
    
    Let me re-read the problem. "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    
    I think I'm misunderstanding the tensor structure. Let me look at the seed program again.
    
    The seed program creates:
    U[i*m+j, c] - shape (n*m, rank) = (6, rank)
    V[j*p+k, c] - shape (m*p, rank) = (18, rank)
    W[k*n+i, c] - shape (p*n, rank) = (12, rank)
    
    And for each (i,j,k):
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    So the tensor being decomposed has indices:
    - First mode: i*m + j, where i in [0,n), j in [0,m) -> 6 values
    - Second mode: j*p + k, where j in [0,m), k in [0,p) -> 18 values
    - Third mode: k*n + i, where k in [0,p), i in [0,n) -> 12 values
    
    And T[i*m+j, j*p+k, k*n+i] = 1 if and only if the indices are consistent (same j and k, etc.)
    
    Actually, looking more carefully: for a given (i,j,k), we set:
    - U[i*m+j, c] = 1
    - V[j*p+k, c] = 1
    - W[k*n+i, c] = 1
    
    So the tensor T has T[a, b, c] = 1 where:
    - a = i*m + j
    - b = j*p + k
    - c = k*n + i
    
    And for each (i,j,k) triple, we have a rank-1 term.
    
    The key constraint is that the same j appears in both a and b, and the same k appears in both b and c, and the same i appears in both c and a.
    
    So T[a, b, c] = 1 if and only if:
    - Let i = a mod m... no wait. a = i*m + j, so j = a % m, i = a // m
    - b = j*p + k, so k = b % p, and we need j = b // p to match
    - c = k*n + i, so i = c % n, and we need k = c // n to match
    
    So T[a, b, c] = 1 if and only if:
    - j_a = a % m
    - i_a = a // m
    - j_b = b // p
    - k_b = b % p
    - k_c = c // n
    - i_c = c % n
    - j_a == j_b and k_b == k_c and i_a == i_c
    
    This is the standard matrix multiplication tensor.
    
    Now, what's the minimal rank? For 2x3 by 3x6, the schoolbook is 2*3*6 = 36.
    
    Can we do better? The rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n) / something... actually, the known bounds are complex.
    
    For small cases, let's think about whether we can exploit structure.
    
    One approach: use the fact that 6 = 2*3 to see if there's a Kronecker product structure.
    
    Actually, let me just try to find a low-rank decomposition using optimization or known results.
    
    For 2x2 by 2x2, the rank is 7 (Strassen's algorithm).
    For 2x3 by 3x2, the rank is... I'm not sure.
    
    Let me try a different approach: use integer programming or a search to find the minimal rank.
    
    Actually, given the time constraint, let me try to construct a decomposition with rank less than 36.
    
    One idea: since 6 = 2*3, maybe we can decompose the 3x6 matrix as a product of smaller matrices.
    
    Let me think about this differently. The tensor T[i*m+j, j*p+k, k*n+i] = 1.
    
    We want to write T as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    For each (i,j,k), we need the sum to be 1, and for all other (a,b,c) not corresponding to a valid (i,j,k), the sum should be 0.
    
    This is a system of linear equations. Let me set it up and try to find a low-rank solution.
    
    Actually, let me just try to use numpy to solve for a decomposition with rank r, for various r.
    
    The number of constraints is 6*18*12 = 1296, but many are zero. The number of non-zero entries is 2*3*6 = 36.
    
    So we have 36 equations (for the non-zero entries) and 1296-36 = 1260 equations (for the zero entries).
    
    Total: 1296 equations.
    
    Unknowns: 6*r + 18*r + 12*r = 36*r.
    
    For r = 36, we have 36*36 = 1296 unknowns, which matches the number of equations. This is the schoolbook solution.
    
    For r < 36, we have fewer unknowns than equations, so the system is overdetermined. We need to find if there's a solution.
    
    Let me try to use least squares or integer programming to find a solution with r < 36.
    
    Actually, since we need exact integer/half-integer entries, let me try a search approach.
    
    Given the time constraint, let me try a few specific ranks and see if I can find a valid decomposition.
    
    Let me start with the schoolbook and try to reduce the rank by merging some terms.
    
    Actually, let me just implement a solver that tries to find a decomposition with rank r, for r from some lower bound up to 36.
    
    The lower bound for the rank of M_{n,m,p} is at least max(n*m, m*p, p*n) / min(n,m,p) or something like that. Actually, a simple lower bound is max(n*m, m*p, p*n) / something... I'm not sure of the exact bound.
    
    For now, let me just try to find a decomposition with rank < 36 using a numerical approach.
    
    Here's my plan:
    1. Set up the tensor T as a 3D array of shape (6, 18, 12).
    2. For a given rank r, try to find U, V, W such that the decomposition is exact.
    3. Use a combination of random initialization and iterative refinement.
    
    But this might be slow. Let me think of a better approach.
    
    Actually, for small tensors, we can use the fact that the rank is the minimum r such that the tensor can be written as a sum of r rank-1 tensors. This is equivalent to finding the minimum r such that there exist U, V, W with the given properties.
    
    One approach is to use alternating least squares or similar methods, but these may not find the global minimum.
    
    Given the time constraint, let me just try the schoolbook decomposition and see if I can improve it slightly.
    
    Actually, let me just return the schoolbook decomposition for now, since I'm not sure I can do better within the time limit.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition". So I need to find the minimum rank.
    
    Let me think about this more carefully. For 2x3 by 3x6, is there a known result?
    
    I recall that for square matrices, the rank of M_{n,n,n} is at least n^2 (trivially) and at most n^3 (schoolbook). For n=2, the rank is 7.
    
    For rectangular matrices, the situation is less clear. Let me just try to find a low-rank decomposition using optimization.
    
    Here's my revised plan:
    1. Use a deadline guard to ensure we finish within the time limit.
    2. Try to find a decomposition with rank r, starting from a lower bound and increasing r.
    3. Use a combination of random search and local optimization.
    
    Let me implement this.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    n, m, p = DIMENSIONS
    # Tensor dimensions
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        # Compute the decomposition
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Outer product
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Lower bound: at least max(dim_a, dim_b, dim_c) / something... actually, a simple lower bound is
    # the rank of the matricization. Let's compute the rank of the matricizations.
    
    # Matricization 1: (dim_a) x (dim_b * dim_c)
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    
    # Matricization 2: (dim_b) x (dim_a * dim_c)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    
    # Matricization 3: (dim_c) x (dim_a * dim_b)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # Try ranks from lower_bound up to best_rank - 1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition with rank r
        # Use a simple approach: try random integer vectors and check
        
        found = False
        attempts = 0
        max_attempts = 10000
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random integer vectors in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_a, r))
            V = np.random.randint(-1, 2, size=(dim_b, r))
            W = np.random.randint(-1, 2, size=(dim_c, r))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
    
    # Convert to lists for the return format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
