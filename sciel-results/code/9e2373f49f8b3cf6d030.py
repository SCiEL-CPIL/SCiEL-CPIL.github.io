"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomp(U, V, W, tol=1e-9):
        """Check if U,V,W with given shapes exactly decompose the tensor."""
        n, m, p = DIMENSIONS
        # Tensor T[i,j,k] = 1 if we consider T as T[(i,j),(j',k),(k',i')] 
        # Actually the tensor for matmul (n,m,p) is: T[i,j,k] where output (i,k) gets contribution from (i,j) and (j,k)
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal 1 when the indices are consistent, 0 otherwise.
        # Actually for matmul, T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k]
        # Wait, no. The tensor T is defined such that <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j]B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the appropriate space)
        # And the decomposition requires: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
        
        # Check: for all i in range(n), j in range(m), k in range(p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: try various ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But realistically for matmul, it's higher.
    
    # For (2,3,9), let's try ranks from 40 to 53
    # Use a randomized approach with structured initialization
    
    best_result = None
    
    # Try structured decompositions
    # Idea: for (2,3,p), we can potentially use rank = 3*p + something
    # Or use the fact that n=2 is small
    
    # Let me try: can we do (2,3,9) with rank 45?
    # One approach: group the p dimension into 3 groups of 3
    # For each group of 3 k-values, do a (2,3,3) multiplication
    # (2,3,3) has known rank... let me think. Schoolbook is 18. 
    # Can we do better? For (2,3,3), I'm not sure of a known improvement.
    
    # Alternative: try to find a decomposition by solving linear equations
    # Fix U and V randomly, solve for W, check if it works
    
    # This is a hard problem. Let me try a few structured approaches.
    
    # Approach 1: Try rank = 48
    # Use the idea that for each j, we have n*p = 18 outputs, and m=3 such j's
    # Total "work" is 3*18 = 54 scalar multiplications
    # Can we save 6 multiplications?
    
    # For now, let's just return schoolbook if we can't find better quickly
    # But let me try one more thing: use numpy to attempt a random search
    
    # Try a few random initializations for lower ranks
    np.random.seed(42)
    for rank in [48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 20:
            break
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This is a system: for fixed (k,i), we have m equations (one per j)
            # and rank unknowns. Underdetermined if rank > m.
            # Let's solve least squares for W given U,V
            
            W = np.zeros((p * n, rank))
            valid = True
            for k in range(p):
                for i in range(n):
                    # For this (k,i), we need: for all j in range(m):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # Then A_mat @ W[k*n+i, :] = ones(m)
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve A_mat @ w = ones(m)
                    try:
                        w, residuals, rank_r, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.linalg.norm(A_mat @ w - np.ones(m)) > 1e-7:
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                if check_decomp(U, V, W):
                    best_result = (U, V, W, rank)
                    break
        if best_result:
            break
    
    if best_result:
        U, V, W, rank = best_result
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

