"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # The matrix multiplication tensor T_{i,j,k} = 1 if we multiply A[i,j] * B[j,k]
    # We want rank-r decomposition: sum_{c=1}^{r} U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # where the product equals 1 for all (i,j,k) in the support (all of them) and 0 otherwise.
    # Actually, the tensor is: for each (i,j,k), the coefficient is 1.
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # And for indices that are NOT valid (i,j,k) combinations... wait, all (i,j,k) are valid.
    # 
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    #
    # This is equivalent to finding a rank-r tensor decomposition of the all-ones tensor of shape (n*m, m*p, p*n)
    # but with the specific index structure.
    #
    # Known results: For 2x3 * 3x4, the minimum rank is known to be 14 (from literature).
    # Let's try to find a rank-14 decomposition.
    #
    # Approach: Use the known structure. The bilinear complexity of 2x3 * 3x4 matrix multiplication.
    # 
    # Let's use a direct search approach with the constraint that entries are integers or half-integers.
    # 
    # First, let's try a rank-14 solution based on known algorithms.
    # 
    # The standard approach: We have 2*3*4 = 24 scalar multiplications in schoolbook.
    # We need to reduce this to 14.
    #
    # Let's set up the problem as finding U, V, W with integer/half-integer entries.
    # 
    # Variables: r columns, each with n*m + m*p + p*n = 6 + 12 + 8 = 26 entries.
    # For r=14, that's 364 variables, each in {0, 0.5, 1, 1.5, ...} or just {-1, -0.5, 0, 0.5, 1, 1.5, 2, ...}
    # 
    # This is a hard combinatorial problem. Let's use a smarter approach.
    #
    # Known: The 2x3 * 3x4 case has bilinear rank 14. Let me try to construct it.
    #
    # Alternative: Use numerical optimization to find a real-valued solution, then round to half-integers.
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Reshape
        U_mat = np.array(U).reshape(n, m, r)
        V_mat = np.array(V).reshape(m, p, r)
        W_mat = np.array(W).reshape(p, n, r)
        # Compute sum over r
        # Result[i, j, k] = sum_c U_mat[i, j, c] * V_mat[j, k, c] * W_mat[k, i, c]
        result = np.zeros((n, m, p))
        for c in range(r):
            result += U_mat[:, :, c][:, :, np.newaxis] * V_mat[:, :, c][np.newaxis, :, :] * W_mat[:, :, c][np.newaxis, np.newaxis, :]
            # Wait, need to be careful with indices.
            # U_mat[i, j, c], V_mat[j, k, c], W_mat[k, i, c]
            # So for fixed c: outer product structure
            # Let me redo this.
        result = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += U_mat[i, j, c] * V_mat[j, k, c] * W_mat[k, i, c]
        return np.allclose(result, np.ones((n, m, p)), atol=1e-6)
    
    def verify_exact(U, V, W):
        """Check exact integer/half-integer decomposition."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        U_mat = np.array(U).reshape(n, m, r)
        V_mat = np.array(V).reshape(m, p, r)
        W_mat = np.array(W).reshape(p, n, r)
        result = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += U_mat[i, j, c] * V_mat[j, k, c] * W_mat[k, i, c]
        # Check if all entries are exactly 1 (within tolerance for floating point)
        return np.allclose(result, np.ones((n, m, p)), atol=1e-9)
    
    # Try known rank-14 solution
    # Let's search for it using a structured approach.
    # 
    # The bilinear rank of 2x3 * 3x4 is 14. Let me try to find it.
    # 
    # I'll use a random restart approach with local optimization.
    
    best_rank = n * m * p  # start with schoolbook
    best_solution = None
    
    # Try ranks from 14 upward
    for target_rank in range(14, n * m * p + 1):
        if time.time() > deadline:
            break
            
        # Try to find a solution of this rank
        found = False
        
        # Use random initialization and local search
        max_attempts = 50 if target_rank <= 16 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization with half-integer values
            np.random.seed(attempt * 1000 + target_rank)
            
            U_rand = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(n * m, target_rank))
            V_rand = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(m * p, target_rank))
            W_rand = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(p * n, target_rank))
            
            if verify_exact(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_solution = (U_rand, V_rand, W_rand)
                found = True
                break
        
        if found:
            break
    
    # If we found a solution, return it
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return schoolbook
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
