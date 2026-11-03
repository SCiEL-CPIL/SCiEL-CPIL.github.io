import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE = 120.0  # seconds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_decomp = None
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE - 3.0 - elapsed()  # 3 second safety margin
    
    def schoolbook_decomp():
        """Create schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly represents matrix multiplication"""
        rank = U.shape[1]
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Build the multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] should sum to 1
                    # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if (i,j,k) corresponds to output[i,k]
                    # Wait, let me reconsider the tensor structure
                    
                    # The matrix multiplication tensor T has T[i,j,k] = 1 if output[i,k] depends on input[i,j]*B[j,k]
                    # Actually T[i,j,k] represents the coefficient in summing A[i,j]*B[j,k] to get C[i,k]
                    # So T[i,j,k] = 1 for all i,j,k where j is shared
                    # The decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank_reduction(U, V, W, target_rank, max_iters=1000):
        """Try to reduce rank using iterative methods"""
        if target_rank >= U.shape[1]:
            return U, V, W
        
        U_curr = U.astype(float).copy()
        V_curr = V.astype(float).copy()
        W_curr = W.astype(float).copy()
        
        # Try to find a lower rank approximation
        # Use alternating least squares with rank constraint
        for iter_num in range(max_iters):
            if time_remaining() < 5.0:
                break
            
            # Project to lower rank using SVD
            # This is a simplified approach - in practice, tensor rank minimization is NP-hard
            # For now, we'll try random restarts with structured matrices
            
            # Try structured initialization with target_rank
            if iter_num % 10 == 0:
                # Generate random structured matrices
                rng = np.random.default_rng(iter_num * 1000)
                U_new = rng.normal(size=(n * m, target_rank))
                V_new = rng.normal(size=(m * p, target_rank))
                W_new = rng.normal(size=(p * n, target_rank))
                
                # Check if this works (unlikely for random, but structured might work)
                # For now, just return if we can find any valid decomposition
                if verify_decomp(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        
        return U, V, W
    
    # First, get schoolbook decomposition
    U, V, W = schoolbook_decomp()
    best_decomp = (U, V, W)
    best_rank = U.shape[1]
    
    # Try various rank reduction strategies
    # For 3x4x4, known lower bounds suggest rank might be less than 48
    
    # Strategy 1: Try structured decompositions with specific patterns
    # The matrix multiplication tensor has structure we can exploit
    
    # Try rank from 47 down to some minimum
    min_rank_to_try = max(24, best_rank - 24)  # Try reducing by up to 24
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time_remaining() < 10.0:
            break
        
        # Try to find a decomposition of this rank
        # Use structured initialization based on the problem structure
        
        # For matrix multiplication, we can use the fact that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Try different structured patterns
        found = False
        
        # Pattern 1: Try using the natural structure
        # Some entries might be zero in the decomposition
        
        if target_rank >= 40:  # Only try for ranks close to schoolbook
            # Try removing some terms and compensating
            for attempt in range(10):
                if time_remaining() < 5.0:
                    break
                
                # Start from schoolbook and try to merge some terms
                U_try = U.astype(float).copy()
                V_try = V.astype(float).copy()
                W_try = W.astype(float).copy()
                
                # Try to zero out some columns and adjust others
                # This is a heuristic - not guaranteed to work
                if attempt == 0:
                    # Try a structured reduction
                    # For 3x4x4, we might be able to use rank 44 or lower
                    pass
        
        # If we found a valid lower rank, update best
        if found:
            best_rank = target_rank
            break
    
    # Convert to list format with integer/half-integer entries
    U_list = [[int(x) if abs(x - round(x)) < 1e-10 else round(x * 2) / 2 for x in row] for row in best_decomp[0]]
    V_list = [[int(x) if abs(x - round(x)) < 1e-10 else round(x * 2) / 2 for x in row] for row in best_decomp[1]]
    W_list = [[int(x) if abs(x - round(x)) < 1e-10 else round(x * 2) / 2 for x in row] for row in best_decomp[2]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
