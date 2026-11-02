"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication.
    
    The tensor T[i,j,k] corresponds to multiplying A[2,4] * B[4,7] = C[2,7].
    T[i,j,k] = 1 if the product involves A[i,j]*B[j,k], else 0.
    Actually, the multiplication tensor has entries T[i,j,k] where i in [0,2), j in [0,4), k in [0,7).
    The standard form: C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is defined such that the matrix multiplication can be recovered as:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (1 if this is a valid term)
    
    Actually, the standard convention: the multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    and the decomposition satisfies:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    
    So U has shape (n*m, rank) = (8, rank)
    V has shape (m*p, rank) = (28, rank)  
    W has shape (p*n, rank) = (14, rank)
    
    And the decomposition must satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a "valid" multiplication term
    
    Actually, for matrix multiplication, the tensor T[i,j,k] = 1 for all (i,j,k) since every term A[i,j]*B[j,k] contributes to C[i,k].
    
    Wait no. The multiplication tensor for multiplying an n x m matrix by an m x p matrix:
    T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 if the j-th column of A and j-th row of B multiply to contribute to C[i,k].
    
    Actually, for standard matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    So the tensor T[i,j,k] = 1 for ALL i, j, k (since every term is used).
    The decomposition requires:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    This is a tensor rank problem. The schoolbook gives rank = n*m*p = 56.
    
    For 2x4x7, we can try to find a lower rank decomposition.
    
    Known results:
    - 2x2x2 has rank 7 (not 8)
    - 2x2x3 has rank 11 (not 12) - actually I'm not sure
    - Small matrix multiplication tensors can have rank below schoolbook
    
    Let me think about this differently. The tensor T of shape (2,4,7) with all entries 1... 
    wait, that's not right either.
    
    Let me reconsider. The multiplication tensor for (n,m,p) has:
    T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] appears in the computation of C[i,k].
    
    For standard matrix multiplication, EVERY term appears, so T[i,j,k] = 1 for all i,j,k.
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,2), j in [0,4), k in [0,7).
    
    This is equivalent to finding a rank-r decomposition of the all-ones tensor of shape (2,4,7)
    under a specific index mapping.
    
    Actually wait - I need to be more careful. The tensor indices are (i,j,k) but they're mapped to:
    U index: i*m+j (so i*4+j)
    V index: j*p+k (so j*7+k)
    W index: k*n+i (so k*2+i)
    
    The constraint is:
    sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*2+i, c] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    
    This is 2*4*7 = 56 equations, and we want to minimize rank r (number of columns in U, V, W).
    
    The schoolbook solution uses r = 56 with U[i*4+j, i*4+j] = 1, V[j*7+k, j*7+k] = 1, W[k*2+i, k*2+i] = 1.
    
    To do better, we need to exploit structure. 
    
    One approach: use the fact that 4 = 2*2 and decompose the middle dimension.
    
    Let me try a different approach: use known low-rank decompositions for small cases and combine them.
    
    Actually, let me try to use a numerical optimization approach. We can set up the problem as:
    Minimize r subject to: sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*2+i, c] = 1 for all (i,j,k)
    
    This is a non-convex problem. But we can try random initialization and gradient descent.
    
    However, the task says entries must be integer/half-integer. So we need exact rational solutions.
    
    Let me try a different strategy: use the structure of the problem.
    
    For 2x2x2, the rank is 7. The decomposition uses 7 terms.
    
    For 2x4x7, one approach is to view it as 2x(2x2)x7 and use tensor product structures.
    
    Actually, let me just try to find a solution with rank less than 56 using optimization.
    
    Given the time constraint, let me try:
    1. Start with schoolbook rank 56
    2. Try to reduce rank by merging terms
    3. Use numerical optimization to find low-rank solutions
    4. Round to nearest half-integers and verify
    
    Let me implement a practical approach:
    - Use random initialization with small rank
    - Use alternating least squares or gradient-based methods
    - Check if we can get exact integer/half-integer solutions
    """
    deadline = time.time() + 140  # Leave 10s margin for the 150s limit
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, r) = (8, r)
    # V shape: (m*p, r) = (28, r)
    # W shape: (p*n, r) = (14, r)
    
    # The constraint: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try various ranks from low to high
    # Schoolbook rank is 56
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def to_half_integers(U, V, W):
        """Try to convert to half-integers."""
        def round_half(x):
            return round(x * 2) / 2
        U_h = [[round_half(x) for x in row] for row in U]
        V_h = [[round_half(x) for x in row] for row in V]
        W_h = [[round_half(x) for x in row] for row in W]
        return U_h, V_h, W_h
    
    # First, try to find a solution with rank < 56
    # Use a simple approach: try to merge schoolbook terms
    
    # Schoolbook decomposition
    rank_school = n * m * p  # 56
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to reduce rank by using the structure
    # Idea: for fixed i, the terms over j,k form a rank-1 structure if we can factor
    
    # Let's try a numerical approach: minimize rank using ALS
    # For a given rank r, we want to find U (8xr), V (28xr), W (14xr)
    # such that sum_c U[i*4+j,c]*V[j*7+k,c]*W[k*2+i,c] = 1
    
    # This can be reformulated: for each (i,j,k), the sum over c must equal 1.
    
    # Let's try random initialization and iterative refinement
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 20 to 55
    import random
    random.seed(42)
    np.random.seed(42)
    
    for r in range(20, min(56, rank_school)):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, r) * 0.1
            V_r = np.random.randn(m*p, r) * 0.1
            W_r = np.random.randn(p*n, r) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*7+k,c] * W[k*2+i,c] = 1 for all k
                # This is a linear system for each (i,j)
                
                for i in range(n):
                    for j in range(m):
                        # For fixed (i,j), we have:
                        # sum_c U[i*m+j,c] * (V[j*7+k,c] * W[k*2+i,c]) = 1 for k=0..p-1
                        # Let A_k,c = V[j*7+k,c] * W[k*2+i,c]
                        # Then sum_c U[i*m+j,c] * A_k,c = 1 for all k
                        # This is: A @ U[i*m+j, :] = 1 (vector of ones)
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V_r[j*p+k, c] * W_r[k*n+i, c]
                        # Solve A @ x = 1
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_r[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k, :] should satisfy
                # sum_c U[i*4+j,c] * V[j*7+k,c] * W[k*2+i,c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U_r[i*m+j, c] * W_r[k*n+i, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V_r[j*p+k, c] = 0  # placeholder
                            V_r[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i, :] should satisfy
                # sum_c U[i*4+j,c] * V[j*7+k,c] * W[k*2+i,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U_r[i*m+j, c] * V_r[j*p+k, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_r[k*n+i, :] = x
                        except:
                            pass
            
            # Check if we found a valid solution
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            # Check numerical validity
            if verify(U_list, V_list, W_list, tol=1e-6):
                # Try to round to half-integers
                U_h, V_h, W_h = to_half_integers(U_list, V_list, W_list)
                if verify(U_h, V_h, W_h, tol=1e-10):
                    best_rank = r
                    best_U, best_V, best_W = U_h, V_h, W_h
                    found = True
                    break
                else:
                    # Accept numerical solution as-is if we can't round
                    # But we need half-integers... so skip
                    pass
        
        if found:
            break
    
    # If we didn't find a better solution, return schoolbook
    if best_rank == rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
