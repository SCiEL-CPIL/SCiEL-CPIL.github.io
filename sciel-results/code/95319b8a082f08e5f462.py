import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _build_constraint_system(n, m, p):
    """Build the constraint structure for matrix multiplication tensor."""
    # For each output (i,k), we need:
    # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = delta(i2==i, j==j2, k2==k)
    # 
    # This is equivalent to: for each c, the outer product W[:,c] ⊗ U[:,c] ⊗ V[:,c]
    # summed over c gives the multiplication tensor.
    #
    # The tensor T has T[i2*m+j, j2*p+k2, k*n+i] = 1 if i2==i and j==j2 and k2==k, else 0
    # But more precisely, for the bilinear complexity formulation:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_{i2,j} U[c,i2*m+j] A[i2,j]) * (sum_{j2,k2} V[c,j2*p+k2] B[j2,k2])
    # For this to equal sum_j A[i,j] B[j,k], we need the trilinear form to match.
    
    return n, m, p

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    
    # For each output position (i, k), check that the bilinear form is correct
    for i in range(n):
        for k in range(p):
            # C[i,k] should = sum_j A[i,j] * B[j,k]
            # The coefficient of A[i2,j] * B[j2,k2] in C[i,k] is:
            # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2]
            # This should be 1 if (i2,j,j2,k2) = (i, j, j, k) for the matching j, else 0
            
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
                            if abs(actual - expected) > tol:
                                return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    
    # Build the full tensor check using einsum-like approach
    # For each (i,k), the matrix of coefficients [i2,j,j2,k2] should be correct
    for i in range(n):
        for k in range(p):
            # Build the n*m x m*p coefficient matrix for this output
            # coeff[i2*m+j, j2*p+k2] = sum_c W[k*n+i,c] * U[c, i2*m+j] * V[c, j2*p+k2]
            coeff = np.zeros((n*m, m*p))
            for c in range(rank):
                w_val = W[k*n+i, c]
                if abs(w_val) > 1e-15:
                    coeff += w_val * np.outer(U[:, c], V[:, c])
            
            # Check: for each j, coeff[i*m+j, j*p+k] should be 1
            for j in range(m):
                if abs(coeff[i*m+j, j*p+k] - 1.0) > tol:
                    return False
            
            # Check: all other entries should be 0
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            if not (i2 == i and j == j2 and k2 == k):
                                if abs(coeff[i2*m+j, j2*p+k2]) > tol:
                                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random initial decomposition."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    return U, V, W

def _als_step(U, V, W, n, m, p, tol=1e-10):
    """One ALS iteration: fix two matrices, solve for the third."""
    rank = U.shape[1]
    
    # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
    # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = target
    # This is a linear system in W[k*n+i, :]
    
    # For fixed (k,i), the equation is:
    # For all (i2, j, j2, k2): sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = T[k*n+i, i2*m+j, j2*p+k2]
    # where T is the target tensor.
    
    # This can be written as: W_row @ (U_col ⊗ V_col) = target
    # where U_col is the column of U indexed by (i2,j) and V_col is the column of V indexed by (j2,k2)
    
    # More precisely, for fixed (k,i):
    # Let A_matrix be of shape (n*m * m*p, rank) where A_matrix[(i2*m+j)*(m*p) + (j2*p+k2), c] = U[c, i2*m+j] * V[c, j2*p+k2]
    # Then W_row @ A_matrix.T = target_vector
    
    # This is expensive to build explicitly. Let's use a different approach.
    
    # Alternative: use the fact that this is a rank-constrained approximation problem.
    # For now, let's just do gradient descent or a simpler approach.
    
    # Actually, let me use a different strategy: since the problem is small, 
    # let me use a direct approach with random restarts and local search.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Start with small ranks and increase
    
    # For 2x3x5 matrix multiplication, known bounds suggest rank around 15-20
    # Let's try ranks from 10 to 29
    
    np.random.seed(42)
    random.seed(42)
    
    for rank in range(10, min(30, best_rank)):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U, V, W = _random_decomposition(n, m, p, rank, seed=42 + restart * 1000)
            
            # Use alternating minimization
            # The objective is to minimize the Frobenius norm of the difference
            # between the computed tensor and the target tensor.
            
            # Let's use a simple approach: iterate by fixing two and doing least squares on the third.
            
            # For each iteration, fix U and V, solve for W via least squares.
            # Then fix U and W, solve for V.
            # Then fix V and W, solve for U.
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W given U, V
                # For each output (i,k), we need:
                # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = T[i,k, i2,j,j2,k2]
                
                # This is a linear system. Let's build it.
                # For fixed (i,k), the unknowns are W[k*n+i, 0], ..., W[k*n+i, rank-1]
                # The equations are: for all (i2, j, j2, k2):
                # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = target[i,k,i2,j,j2,k2]
                
                # Build the system for one (i,k) at a time
                W_new = np.zeros((p*n, rank))
                for i in range(n):
                    for k in range(p):
                        # Build A matrix: shape (n*m*m*p, rank)
                        # A[(i2*m+j)*(m*p) + (j2*p+k2), c] = U[c, i2*m+j] * V[c, j2*p+k2]
                        A = np.zeros((n*m * m*p, rank))
                        b = np.zeros(n*m * m*p)
                        
                        for i2 in range(n):
                            for j in range(m):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        idx = (i2*m + j) * (m*p) + (j2*p + k2)
                                        # A[idx, c] = U[c, i2*m+j] * V[c, j2*p+k2]
                                        A[idx, :] = U[:, i2*m+j] * V[:, j2*p+k2]
                                        # Target: 1 if i2==i, j==j2, k2==k, else 0
                                        b[idx] = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                        
                        # Solve A @ W_row = b
                        # W_row = W[k*n+i, :]
                        try:
                            W_row, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k*n+i, :] = W_row
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                
                W = W_new
                
                # Solve for V given U, W
                V_new = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        # For each (j,k), the unknowns are V[j*p+k, 0], ..., V[j*p+k, rank-1]
                        # Equations: for all (i2, j2, k2, i):
                        # sum_c U[c, i2*m+j2] * V[j*p+k, c] * W[k*n+i, c] = target[i,k,i2,j2,k]
                        # Wait, I need to be more careful.
                        
                        # The condition is: for all (i, k, i2, j, j2, k2):
                        # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j2*p+k2] = delta(i2==i, j==j2, k2==k)
                        
                        # If we fix U and W, and want to solve for V[j*p+k, :],
                        # The relevant equations are those where the V index is (j2*p+k2) = (j*p+k)
                        # So j2=j, k2=k.
                        # For all (i, i2, j2, k2) with j2=j, k2=k:
                        # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j*p+k] = delta(i2==i, j==j, k==k)
                        # = delta(i2==i) * 1 * 1 if i2==i else 0
                        # Wait, the delta is delta(i2==i, j==j2, k2==k) = delta(i2==i, j==j, k==k) = delta(i2==i)
                        
                        # So for fixed (j,k), the equations are for all (i, i2):
                        # sum_c W[k*n+i, c] * U[c, i2*m+j] * V[c, j*p+k] = 1 if i2==i else 0
                        
                        # This is a system with n*n equations and rank unknowns.
                        A = np.zeros((n*n, rank))
                        b = np.zeros(n*n)
                        
                        for i in range(n):
                            for i2 in range(n):
                                idx = i * n + i2
                                A[idx, :] = W[k*n+i, :] * U[:, i2*m+j]
                                b[idx] = 1.0 if (i2 == i) else 0.0
                        
                        try:
                            V_row, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p+k, :] = V_row
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                
                V = V_new
                
                # Solve for U given V, W
                U_new = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        # For each (i,j), unknowns are U[i*m+j, 0], ..., U[i*m+j, rank-1]
                        # Equations: for all (k, i2, j2, k2) with i2=i, j=j2:
                        # sum_c W[k*n+i2, c] * U[c, i*m+j] * V[c, j*p+k2] = delta(i2==i, j==j, k2==k)
                        # = delta(k2==k) since i2=i, j=j
                        
                        # So for fixed (i,j), equations for all (k, k2):
                        # sum_c W[k*n+i, c] * U[c, i*m+j] * V[c, j*p+k2] = 1 if k2==k else 0
                        
                        A = np.zeros((p*p, rank))
                        b = np.zeros(p*p)
                        
                        for k in range(p):
                            for k2 in range(p):
                                idx = k * p + k2
                                A[idx, :] = W[k*n+i, :] * V[:, j*p+k2]
                                b[idx] = 1.0 if (k2 == k) else 0.0
                        
                        try:
                            U_row, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m+j, :] = U_row
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                
                U = U_new
                
                # Check convergence
                if iteration % 10 == 9:
                    if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-7):
                        converged = True
                        break
            
            if converged:
                # Verify with tighter tolerance
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
                    break  # Found a valid decomposition for this rank, try lower
    
    if time.time() < deadline:
        # Try to polish the best solution found
        if best_rank < rank_school:
            # Try to verify with very tight tolerance and potentially round to half-integers
            pass
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
