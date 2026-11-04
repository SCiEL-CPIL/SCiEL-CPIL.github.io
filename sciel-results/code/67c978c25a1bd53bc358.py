"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x5 by 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # The tensor decomposition condition:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Schoolbook solution has rank n*m*p = 100 with U[i*m+j, i*m*p+j*p+k] = 1 etc.
    #
    # Key insight: We can use the structure of matrix multiplication to find
    # lower-rank decompositions. For small matrices, known results suggest
    # that rank can be reduced below schoolbook.
    #
    # Strategy: Use a structured approach based on the fact that:
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as: C = A * B
    # Using the identity: sum_j A[i,j] B[j,k] = (A ⊗ B)[i,k]
    #
    # A better approach: Use the standard matrix multiplication tensor
    # and try to find a low-rank decomposition via:
    # 1. Start with schoolbook
    # 2. Try to merge/eliminate terms using algebraic identities
    #
    # For 4x5x5, let's try a systematic search using the structure.
    
    # Build schoolbook solution
    def make_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return U, V, W
    
    U_school, V_school, W_school = make_schoolbook()
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # Approach: Use the fact that matrix multiplication can be expressed
    # using fewer terms through clever algebraic manipulation.
    #
    # For example, for 2x2x2, Strassen's algorithm achieves rank 7 instead of 8.
    # For larger matrices, similar techniques can be applied.
    #
    # Let's try a greedy approach: start with schoolbook and try to eliminate
    # one term at a time by expressing it as a combination of others.
    
    best_rank = schoolbook_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try rank reduction from 99 down
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 30, 0), -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # Method: Start from schoolbook and try to zero out one column
        # while adjusting others to maintain the tensor equation.
        
        success = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook but with one column removed
            # and try to fix the rest
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Copy first target_rank columns from schoolbook
            U[:, :target_rank] = U_school[:, :target_rank]
            V[:, :target_rank] = V_school[:, :target_rank]
            W[:, :target_rank] = W_school[:, :target_rank]
            
            # Now try to fix the tensor equation by adjusting the last column
            # or by redistributing the missing term
            #
            # The missing term corresponds to column schoolbook_rank-1
            # which was (i=n-1, j=m-1, k=p-1) in schoolbook
            #
            # We need: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k
            #
            # With the last column removed, the term for (n-1, m-1, p-1) is missing.
            # We need to add it back by adjusting other terms.
            
            # Compute current error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        e = abs(s - 1.0)
                        if e > max_err:
                            max_err = e
            
            if max_err < 1e-9:
                success = True
                break
            
            # Try to fix by adjusting the last column
            # The missing contribution is at (i=n-1, j=m-1, k=p-1)
            # We need to add 1 to that position's sum
            #
            # One approach: set U[n-1*m+m-1, -1] = a, V[(m-1)*p+p-1, -1] = b, 
            # W[(p-1)*n+n-1, -1] = c such that a*b*c = 1
            # But this might break other positions.
            
            # Alternative: use a more systematic approach
            # Try random perturbations of the last few columns
            
            if attempt > 0:
                # Add some randomness
                np.random.seed(attempt * 42)
                U[:, -3:] += np.random.randn(n * m, 3) * 0.1
                V[:, -3:] += np.random.randn(m * p, 3) * 0.1
                W[:, -3:] += np.random.randn(p * n, 3) * 0.1
            
            # Iterate to fix errors
            for it in range(20):
                if time.time() > deadline:
                    break
                
                # Find the worst error
                max_err = 0.0
                err_i, err_j, err_k = 0, 0, 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            e = abs(s - 1.0)
                            if e > max_err:
                                max_err = e
                                err_i, err_j, err_k = i, j, k
                
                if max_err < 1e-9:
                    success = True
                    break
                
                # Adjust the last column to fix the error
                i, j, k = err_i, err_j, err_k
                current = 0.0
                for r in range(target_rank):
                    current += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                needed = 1.0 - current
                
                # Try to fix by adjusting U[i*m+j, -1]
                if abs(V[j*p+k, -1] * W[k*n+i, -1]) > 1e-12:
                    U[i*m+j, -1] += needed / (V[j*p+k, -1] * W[k*n+i, -1])
                elif abs(U[i*m+j, -1] * W[k*n+i, -1]) > 1e-12:
                    V[j*p+k, -1] += needed / (U[i*m+j, -1] * W[k*n+i, -1])
                elif abs(U[i*m+j, -1] * V[j*p+k, -1]) > 1e-12:
                    W[k*n+i, -1] += needed / (U[i*m+j, -1] * V[j*p+k, -1])
            
            if success:
                break
        
        if success:
            if verify(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                # Continue trying to reduce further
                continue
            else:
                break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
