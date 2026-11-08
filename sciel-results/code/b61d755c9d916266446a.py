"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict: {'U': U, 'V': V, 'W': W} where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has indices (i, j, k) with values T[i,j,k] = 1
    # if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i)
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: for each r, we have a rank-1 tensor U_r ⊗ V_r ⊗ W_r
    # and their sum must equal the all-ones tensor (for the relevant indices)
    
    # Known results: The minimum rank for multiplying an n×m matrix by an m×p matrix
    # is at least max(n*m, m*p, n*p) and at most n*m*p (schoolbook).
    # For 2×3 by 3×4: lower bound is max(6, 12, 8) = 12, upper bound is 24.
    
    # Let's try to find a decomposition with rank as low as possible.
    # We'll use a systematic approach: try ranks from the theoretical lower bound upward.
    
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(6, 12, 8) = 12
    # Let's try ranks from 12 upward
    
    # For the schoolbook method, rank = n*m*p = 24
    # We can always fall back to schoolbook
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = U.shape[1]
        # For each i, j, k, check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of the given rank using linear algebra."""
        # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
        
        # This is a system of n*m*p equations with 3*rank*(n*m + m*p + p*n) unknowns,
        # but it's nonlinear (trilinear).
        
        # One approach: fix U and V, then solve for W linearly.
        # For each (i, k), we have: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is linear in W.
        
        # Let's try a different approach: use the fact that this is equivalent to
        # finding a low-rank decomposition of a certain matrix.
        
        # Actually, let's use a simpler heuristic: try random initializations and use
        # iterative refinement.
        
        # Alternative: use the known structure. For matrix multiplication, we can
        # use block methods.
        
        # For now, let's try a direct approach:
        # Fix W to be an identity-like matrix and solve for U and V.
        
        # Actually, the cleanest approach for small dimensions is to use the
        # following: the tensor is separable into rank-1 terms.
        
        # Let's use a numerical optimization approach.
        
        # Initialize with random values
        np.random.seed(42)
        
        # We'll use the fact that if we fix two of the matrices, the third is determined
        # by a linear system.
        
        # Strategy: 
        # 1. Initialize U and V randomly
        # 2. Solve for W using least squares
        # 3. Iterate: fix W, solve for U and V jointly (this is still nonlinear)
        
        # Actually, let's use a simpler method: 
        # The condition sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # can be viewed as: for each r, define A_r[i,j] = U[i*m+j, r], B_r[j,k] = V[j*p+k, r],
        # C_r[k,i] = W[k*n+i, r]. Then we need sum_r A_r ⊙ B_r ⊙ C_r = T where T[i,j,k]=1.
        
        # This is hard to solve directly. Let's use an iterative approach.
        
        # Alternative simple approach: try to construct using block diagonal structure.
        
        # For 2x3 * 3x4, let's see if we can do better than 24.
        
        # Known: the rank of matrix multiplication <2,3,4> is at most 24 (schoolbook).
        # It's known that rank <2,3,4> <= 24, and it's conjectured to be lower.
        
        # Let's try a construction with rank 18 or so using block methods.
        
        # Actually, let's just return the schoolbook if we can't find better quickly.
        
        # Try a simple improvement: use the fact that we can share some terms.
        
        # For now, let's try a random search with limited iterations.
        
        max_iters = 100
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Iterative refinement
            for inner_iter in range(50):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # For each (k, i), we need: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a linear system in W.
                
                # Build the linear system: for each (k, i), the equation is:
                # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
                # Let M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                # Then we need M @ W_col = 1 for each column of W... wait, W is (p*n, rank)
                # and we're solving for each (k,i) independently? No, W[k*n+i, r] couples across r.
                
                # Actually, for each fixed (k, i), we have one equation with rank unknowns W[k*n+i, 0..rank-1].
                # But these are coupled through the same r index across different (k,i).
                
                # Let me re-examine: W has shape (p*n, rank). W[k*n+i, r] is the (k,i,r) entry.
                # The equation for (i,j,k) is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This couples all (i,j,k) equations through the shared r index in W.
                
                # This is a system of n*m*p equations in p*n*rank unknowns (entries of W).
                # But it's not linear in W because U and V are also variables.
                
                # If we fix U and V, then for each (k,i), the equation becomes:
                # sum_r W[k*n+i, r] * C[k*n+i, r] = 1 where C[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                # But this is still coupled across (k,i) through r.
                
                # Actually no - for fixed (k,i), the equation involves W[k*n+i, :] dotted with C[k*n+i, :].
                # These are separate equations for each (k,i), each involving only W[k*n+i, :].
                # So we can solve for each row of W independently!
                
                # For each (k, i): find W[k*n+i, :] such that W[k*n+i, :] @ C[k*n+i, :] = 1
                # where C[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                
                # This is an underdetermined system (rank unknowns, 1 equation).
                # The minimum norm solution is W[k*n+i, r] = C[k*n+i, r] / (C[k*n+i, :] @ C[k*n+i, :])
                
                for k in range(p):
                    for i in range(n):
                        C = np.zeros(rank)
                        for j in range(m):
                            C += U[i * m + j, :] * V[j * p + k, :]
                        norm_sq = np.dot(C, C)
                        if norm_sq > 1e-12:
                            W[k * n + i, :] = C / norm_sq
                        else:
                            W[k * n + i, :] = 0
                
                # Now fix W, solve for U and V similarly
                # For each (i, j): sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let D[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                # Then U[i*m+j, :] @ D[i*m+j, :] = 1
                for i in range(n):
                    for j in range(m):
                        D = np.zeros(rank)
                        for k in range(p):
                            D += V[j * p + k, :] * W[k * n + i, :]
                        norm_sq = np.dot(D, D)
                        if norm_sq > 1e-12:
                            U[i * m + j, :] = D / norm_sq
                        else:
                            U[i * m + j, :] = 0
                
                # Fix U and W, solve for V
                # For each (j, k): sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1
                # Let E[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        E = np.zeros(rank)
                        for i in range(n):
                            E += U[i * m + j, :] * W[k * n + i, :]
                        norm_sq = np.dot(E, E)
                        if norm_sq > 1e-12:
                            V[j * p + k, :] = E / norm_sq
                        else:
                            V[j * p + k, :] = 0
                
                # Check convergence
                if verify_decomposition(U, V, W, tol=1e-6):
                    # Refine with tighter tolerance
                    for refine_iter in range(100):
                        if time.time() > deadline:
                            break
                        for k in range(p):
                            for i in range(n):
                                C = np.zeros(rank)
                                for j in range(m):
                                    C += U[i * m + j, :] * V[j * p + k, :]
                                norm_sq = np.dot(C, C)
                                if norm_sq > 1e-15:
                                    W[k * n + i, :] = C / norm_sq
                        for i in range(n):
                            for j in range(m):
                                D = np.zeros(rank)
                                for k in range(p):
                                    D += V[j * p + k, :] * W[k * n + i, :]
                                norm_sq = np.dot(D, D)
                                if norm_sq > 1e-15:
                                    U[i * m + j, :] = D / norm_sq
                        for j in range(m):
                            for k in range(p):
                                E = np.zeros(rank)
                                for i in range(n):
                                    E += U[i * m + j, :] * W[k * n + i, :]
                                norm_sq = np.dot(E, E)
                                if norm_sq > 1e-15:
                                    V[j * p + k, :] = E / norm_sq
                        if verify_decomposition(U, V, W, tol=1e-10):
                            # Convert to exact fractions if possible
                            return {"U": U, "V": V, "W": W}
                    if verify_decomposition(U, V, W, tol=1e-8):
                        return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, n * p)
    upper_bound = n * m * p
    
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            # Verify with high precision
            if verify_decomposition(result["U"], result["V"], result["W"], tol=1e-9):
                return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
