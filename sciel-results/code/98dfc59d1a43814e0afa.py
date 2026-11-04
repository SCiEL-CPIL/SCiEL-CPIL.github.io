"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try to find a low-rank decomposition
    # The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k) in the tensor sense
    # Actually the tensor is: T[i,j,k] where the multiplication is sum_j A[i,j]*B[j,k]
    # The tensor entries are: T[i,j,k] = e_i ⊗ e_j ⊗ e_k for the bilinear form
    
    # The condition is: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should 
    # equal 1 if (i,j,k) corresponds to a term in the sum, else 0.
    # More precisely, the multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # and the decomposition must satisfy: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Wait, that's not right. The tensor is defined such that the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has components T[i,j,k] and the rank-1 decomposition gives:
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # And we need T[i,j,k] = 1 for all valid (i,j,k) since each term contributes.
    
    # Actually, the standard formulation: the matrix multiplication tensor M_{n,m,p} is defined
    # as sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k (in the appropriate spaces)
    # A rank-1 decomposition: M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c in R^n, v_c in R^m, w_c in R^p
    # The factor matrices are: U[i*m+j, c] = u_c[i] * v_c[j] ... no wait.
    
    # Let me re-read the interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition condition is: for all i in [n], j in [m], k in [p]:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is because the (i,j,k) entry of the multiplication tensor is 1.
    
    # So we need to find U, V, W of rank r such that for all (i,j,k):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p equations. Let's try to solve it.
    
    # For a given rank r, we have n*m*r + m*p*r + p*n*r unknowns and n*m*p equations.
    # We need n*m*r + m*p*r + p*n*r >= n*m*p for a chance of solution.
    # For n=3,m=4,p=5: 12r + 20r + 15r = 47r >= 60, so r >= 2.
    # But this is a necessary condition, not sufficient. The equations are nonlinear (trilinear).
    
    # Strategy: Try various ranks and use optimization to find a solution.
    # Start with a reasonable rank and try to minimize the error.
    
    best_result = None
    best_rank = n * m * p  # fallback: schoolbook
    
    # Generate schoolbook as fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n*m*p
    
    # Try to find better decompositions
    # Use a simple approach: for each candidate rank, try random initialization and 
    # use a fixed-point or gradient-based method
    
    def compute_error(U, V, W, rank):
        """Compute the maximum error in the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(rank, max_iters=500):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        
        # Simple iterative method: alternate updates
        # For fixed V, W: U[i*m+j,c] should satisfy sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is hard to solve directly. Use a simple gradient descent on the error.
        
        # Alternative: use the fact that if we fix two matrices, the third is determined
        # by a linear system. But the system is over-determined.
        
        # Let's try a different approach: use known constructions for small matrices.
        # For 3x4x5, try rank 45, 44, 43, ...
        
        return None  # Placeholder
    
    # For small dimensions, let's use a known result or try specific constructions.
    # The rank of M_{3,4,5} is at most 3*4*5 = 60, but likely much less.
    # A simple improvement: note that for fixed j, the terms involving j are independent.
    # We can group by j: for each j, we have a 3x5 multiplication tensor M_{3,5}.
    # The rank of M_{3,5} is at most 15 (schoolbook) but can be less.
    # Actually, the full tensor is sum_j (e_j ⊗ M_{3,5} ⊗ e_j) ... no, that's not quite right.
    
    # Let me just return the schoolbook for now, but try a few quick optimizations.
    # Given time constraints, return the schoolbook.
    
    return best_result
# EVOLVE-BLOCK-END
