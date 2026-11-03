"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 105
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using structured search
    # The matrix multiplication tensor has rank at most n*m*p but potentially lower.
    # For small matrices, we can use bilinear algorithms.
    
    # Strategy: Use the Strassen-like approach or direct rank minimization
    # For 3x5*5x7, let's try a few structured approaches
    
    # Approach 1: Try rank slightly below schoolbook using random initialization + optimization
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this is the (i,j) element of A times (j,k) of B contributing to C[i,k]
    # Actually, the matrix multiplication tensor T satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the bilinear map)
    # Wait, let me reconsider. The matrix multiplication tensor is defined such that:
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # T = sum_r U_r ⊗ V_r ⊗ W_r where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Hmm, that's not quite right either. Let me think more carefully.
    
    # The standard formulation: We want to compute C = A * B where A is n×m, B is m×p, C is n×p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The bilinear algorithm uses rank r such that:
    # C[i,k] = sum_{r=0}^{r-1} W[r, i, k] * (sum_{j} U[r, i, j] * A[i,j]) * (sum_{j} V[r, j, k] * B[j,k])
    
    # In the factor matrix convention:
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]  
    # W has shape (p*n, rank): W[k*n + i, r]
    
    # The decomposition satisfies: for all i, k:
    # sum_{j=0}^{m-1} A[i,j] * B[j,k] = sum_{r=0}^{rank-1} W[k*n+i, r] * (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k])
    
    # This must hold for all A, B, so we need:
    # For all i, j, k: the coefficient of A[i,j]*B[j,k] on LHS is 1
    # On RHS: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all i, j, k
    
    # So the constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 105 equations in rank^3 variables (but with structure).
    
    # Let's try to find a solution with rank < 105 using numerical optimization.
    
    def try_rank(rank, max_iter=500, seed=42):
        """Try to find a decomposition of given rank using gradient-based optimization."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # We want: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a non-convex problem. Let's use a simple iterative approach.
        
        # Alternative: use the fact that this is a tensor completion problem.
        # Let's try a different approach: start from schoolbook and try to merge terms.
        
        # For now, let's just return None if we can't find a better solution quickly.
        return None
    
    # Approach 2: Use known results for small matrix multiplication
    # For 2x2*2x2, Strassen gives rank 7 < 8
    # For our case 3x5*5x7, we need to search.
    
    # Let's try a direct approach: assume the decomposition has a certain structure
    # and solve for the entries.
    
    # Given the complexity, let's try a few random seeds with a simple heuristic:
    # Start from schoolbook and try to reduce rank by finding linear dependencies.
    
    # Actually, let's try a more systematic approach using the structure of the problem.
    # The key insight is that the tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor 
    # in the tensor product space, but the constraint is on the bilinear form.
    
    # Let me try a different strategy: use the fact that for fixed j, the sub-tensor 
    # T[i,j,k] for all i,k is rank-1 (it's all ones). So we can potentially share 
    # factors across different j values.
    
    # For each j, we have a n×p matrix of ones. The bilinear algorithm for computing 
    # sum_j A[i,j] B[j,k] can be seen as summing m rank-1 tensors.
    
    # A simple lower bound: the rank is at least max(n*p, n*m, m*p) = max(21, 15, 35) = 35
    # because the tensor must have certain slice ranks.
    
    # Let's try rank 35 and see if we can find a solution.
    
    def solve_for_rank(rank, num_trials=5, time_per_trial=20):
        """Try to find a decomposition of given rank."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            seed = 42 + trial * 7
            rng = np.random.RandomState(seed)
            
            # Initialize with structured random values
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Normalize to avoid numerical issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Use a simple iterative refinement
            # We want to minimize: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            
            # This is a hard non-convex problem. Let's use a simple gradient descent.
            # But this might be too slow. Let's try a different approach.
            
            # Alternative: use the fact that we can set up a linear system if we fix two of the three matrices.
            # Fix U and V, then W is determined by a linear system.
            
            # For each (i,k), we have: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
            # Wait, that's not right. The constraint is:
            # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # If we fix U and V, then for each (i,k), we have:
            # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
            # No, that's not correct either. Let me re-derive.
            
            # The constraint is: for all i, j, k:
            # sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a trilinear equation. If we fix U and V, it's linear in W, but 
            # the coefficient of W[k*n+i, r] depends on i, j, k, r.
            
            # Actually, for fixed i, k, the equations for different j are:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each j
            
            # This is a system of m equations in rank unknowns (the W[k*n+i, r] for r=0..rank-1).
            # If rank >= m, this system might have a solution.
            
            # Let's try: fix U and V, and for each (i,k), solve for W[k*n+i, :].
            
            # For each i in range(n), k in range(p):
            #   Build matrix M_{j,r} = U[i*m+j, r] * V[j*p+k, r] for j=0..m-1, r=0..rank-1
            #   Solve M * W[k*n+i, :] = ones(m)
            
            # This is a linear least squares problem. If rank >= m, we can find a solution.
            
            if rank < m:
                continue
            
            success = True
            for i in range(n):
                for k in range(p):
                    if time.time() > deadline:
                        return None
                    # Build the coefficient matrix
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    
                    # Solve M * w = ones(m)
                    try:
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if the solution is exact
                        if np.max(np.abs(M @ w - np.ones(m))) < 1e-8:
                            W[k * n + i, :] = w
                        else:
                            # The system is inconsistent; this initialization doesn't work
                            success = False
                            break
                    except:
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # Verify the solution
                if verify(U, V, W, rank):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    # Try ranks from a lower bound up to schoolbook
    lower_bound = max(n * p, n * m, m * p)  # 35
    
    # Try a few ranks near the lower bound
    for rank in range(lower_bound, min(lower_bound + 20, rank_schoolbook + 1)):
        if time.time() > deadline:
            break
        
        result = solve_for_rank(rank, num_trials=3, time_per_trial=15)
        if result is not None:
            # Round to nearest half-integer if possible
            result = round_to_half_integer(result)
            if verify(result["U"], result["V"], result["W"], rank):
                return result
    
    # If we couldn't find a better solution, return schoolbook
    return schoolbook()

def round_to_half_integer(result):
    """Try to round entries to nearest half-integer and verify."""
    U = np.round(result["U"] * 2) / 2
    V = np.round(result["V"] * 2) / 2
    W = np.round(result["W"] * 2) / 2
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > 1e-6:
                    return result  # Return unrounded if rounding breaks the solution
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
