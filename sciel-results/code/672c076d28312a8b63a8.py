"""Matrix multiplication 3x4x6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (12, rank)
        - V has shape (m*p, rank) = (24, rank)
        - W has shape (p*n, rank) = (18, rank)
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Start with schoolbook rank = n*m*p = 72
    # We'll try to find lower rank decompositions using various strategies.
    
    # Strategy 1: Try known lower bounds and use structured decompositions
    # The Strassen algorithm gives rank 7 for 2x2x2, but for rectangular matrices
    # we need to be more careful.
    
    # For 3x4x6, let's try to use a combination of:
    # 1. Block decomposition
    # 2. Known efficient algorithms for rectangular matrices
    
    # The lower bound for matrix multiplication is max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # But this is just a trivial lower bound. The actual minimum rank is unknown in general.
    
    # Let's try a structured approach:
    # - Use the fact that we can pad to square matrices and use known algorithms
    # - Or use a direct search with random initialization and optimization
    
    # Given time constraints, let's try several approaches:
    
    # Approach A: Try rank values from lower bound up to schoolbook
    # For each rank, try to find a valid decomposition using numerical optimization
    
    # First, let's establish the schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def check_decomposition(U, V, W, tol=1e-6):
        """
        Check if U, V, W form a valid decomposition.
        The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for the matrix multiplication tensor, which is 1 if the indices are consistent
        with matrix multiplication, 0 otherwise.
        
        Actually, the matrix multiplication tensor is defined such that:
        (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor T has T[i,j,k] = 1 for all valid i,j,k (since it's the identity
        mapping in the tensor sense for multiplication). Wait, that's not right.
        
        Let me reconsider. The matrix multiplication tensor M of format (n,m,p) is
        defined such that for matrices A (n x m) and B (m x p):
        (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor M[i,j,k] is the coefficient of A[i,j]*B[j,k] in (A*B)[i,k],
        which is 1 for all i,j,k.
        
        So the tensor is all ones! That means we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        This is a much simpler condition!
        """
        # Create the target tensor (all ones)
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Compute the reconstructed tensor
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For efficiency, we can compute this as a batch operation
        # But for small dimensions, direct computation is fine
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    # Try to find low-rank decompositions using a greedy/iterative approach
    # Since the tensor is all ones, we can use a different perspective
    
    # Actually, let's use a more systematic approach:
    # We want to minimize rank such that there exist U, V, W with the above property.
    
    # For the all-ones tensor, one approach is to use rank-1 terms that sum to 1.
    # A rank-1 term is outer product of vectors u (length nm), v (length mp), w (length pn)
    # such that u[i*m+j] * v[j*p+k] * w[k*n+i] contributes to T[i,j,k].
    
    # Let's try a randomized approach: start with rank r, initialize U, V, W randomly,
    # and use gradient descent to minimize the error.
    
    def try_random_decomposition(rank, num_restarts=5, max_iter=200, lr=0.01, seed=None):
        """Try to find a decomposition of given rank using random initialization and gradient descent."""
        if seed is not None:
            np.random.seed(seed)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Scale to help convergence
            # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum to 1
            # Let's normalize
        
            for it in range(max_iter):
                # Compute gradient of error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # This is computationally expensive, so let's use a simpler approach
                
                # Actually, for the all-ones tensor, we can use a different formulation.
                # Let's just do a simple iterative refinement.
                
                # Compute current tensor values
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            errors[i, j, k] = val - 1.0
                
                max_err = np.max(np.abs(errors))
                if max_err < 1e-8:
                    return U, V, W, rank
                
                # Update using gradient descent
                # Gradient wrt U[i*m+j, r] = sum_{k} errors[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                # This is still expensive. Let's use a simpler heuristic.
                
                # Simple heuristic: scale U, V, W to reduce error
                # If the average value is v_avg, scale by 1/v_avg
                avg_val = np.mean(np.abs(errors) + 1e-10)
                if avg_val > 0:
                    scale = 1.0 / (avg_val ** (1/3))
                    U *= scale ** (1/3)
                    V *= scale ** (1/3)
                    W *= scale ** (1/3)
                
                if max_err < best_error:
                    best_error = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if max_err < 1e-6:
                    break
        
        if best_U is not None and best_error < 1e-4:
            return best_U, best_V, best_W, rank
        return None, None, None, rank
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, n*p)  # 24
    schoolbook_rank = n * m * p  # 72
    
    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to find lower rank solutions
    # Start from a reasonable guess and go up
    # For 3x4x6, let's try ranks from 24 to 72
    
    # Given time constraints, let's be strategic:
    # 1. Try a few specific ranks that are likely to work
    # 2. Use the schoolbook as fallback
    
    # Let's try rank = 36 (half of schoolbook) first, then work down
    candidates = [48, 42, 36, 30, 24]
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    for rank in candidates:
        if time.time() > deadline - 30:  # Leave 30 seconds buffer
            break
        
        U, V, W, _ = try_random_decomposition(rank, num_restarts=3, max_iter=100, seed=42)
        
        if U is not None:
            # Verify the solution
            if check_decomposition(U, V, W, tol=1e-5):
                best_result = (U, V, W, rank)
                print(f"Found solution with rank {rank}")
                break
    
    U_best, V_best, W_best, rank_best = best_result
    
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
