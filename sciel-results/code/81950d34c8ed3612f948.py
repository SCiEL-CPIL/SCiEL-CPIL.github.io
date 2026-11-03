"""Matrix multiplication tensor decomposition for 4x4 * 4x6.
Finds a rank-minimal exact tensor decomposition with integer/half-integer entries.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (24, rank)
    - W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    And for all other (i,j,k) combinations where the product should be 0,
    the sum equals 0.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 96
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find a lower rank decomposition using structured approaches
    # For 4x4 * 4x6, we can try rank = 80, 72, etc.
    
    # Approach: Use a Kronecker-product-based decomposition
    # If we can decompose the 4x4 and 4x6 multiplications separately and combine them
    # But that's not directly applicable.
    
    # Alternative: Try to find a decomposition with rank < 96 using random initialization
    # and iterative refinement (like tensor train decomposition)
    
    # Let's try a direct approach: assume rank r, and try to solve for U, V, W
    # The constraint is that for each (i,j,k) with i in [0,4), j in [0,4), k in [0,6):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "invalid" combinations (which don't exist in standard matmul), we need 0.
    
    # Actually, in standard matrix multiplication, the tensor T[i,j,k] = 1 if and only if
    # we're computing C[i,k] = sum_j A[i,j] * B[j,k]. So the tensor has 1s at positions
    # (i,j,k) for all valid i,j,k, and 0 elsewhere. But since all i,j,k are valid in their
    # ranges, the tensor is all 1s on the support.
    
    # Wait, let me re-read the problem. The matrix multiplication tensor for (n,m,p) has
    # entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p). There are no "zero"
    # entries in the tensor itself - the zeros come from the structure of how A and B are
    # indexed.
    
    # So we need: for all i,j,k in valid ranges, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 96 equations. We want to minimize rank r.
    
    # Let's try rank = 80 and see if we can find a solution
    # We'll use a structured approach: try to find U, V, W with small integer entries
    
    # Try a parametric approach with rank = 80
    # Use a grid search over possible small integer values
    
    # Actually, let's try a simpler approach: use the fact that we can factor the problem
    # into smaller subproblems.
    
    # For 4x4 * 4x6, we can think of it as two 4x4 * 4x3 multiplications
    # Or we can use a block structure.
    
    # Let's try rank = 80 with a randomized approach
    def try_rank(r, max_iter=1000, seed=42):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        
        # Initialize with random small integers
        U = np.random.randint(-1, 2, (n*m, r))
        V = np.random.randint(-1, 2, (m*p, r))
        W = np.random.randint(-1, 2, (p*n, r))
        
        # Compute the current tensor values
        def compute_tensor(U, V, W):
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            return T
        
        T_target = np.ones((n, m, p))
        
        # Iteratively adjust entries to minimize error
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
                
            T = compute_tensor(U, V, W)
            error = np.sum((T - T_target)**2)
            
            if error < 1e-10:
                return U, V, W, True
            
            # Find the entry with largest error contribution
            max_error_idx = np.unravel_index(np.argmax(np.abs(T - T_target)), T.shape)
            i, j, k = max_error_idx
            diff = T[i,j,k] - 1
            
            # Try adjusting U[i*m+j, :], V[j*p+k, :], or W[k*n+i, :]
            # Pick the one with largest norm
            u_idx = i*m+j
            v_idx = j*p+k
            w_idx = k*n+i
            
            # Try all three options
            best_adjustment = None
            best_new_error = error
            
            for factor, idx in [(U, u_idx), (V, v_idx), (W, w_idx)]:
                for c in range(r):
                    # Try changing factor[idx, c] by +/- 1
                    for delta in [-1, 1]:
                        old_val = factor[idx, c]
                        factor[idx, c] = old_val + delta
                        T_new = compute_tensor(U, V, W)
                        new_error = np.sum((T_new - T_target)**2)
                        factor[idx, c] = old_val
                        if new_error < best_new_error:
                            best_new_error = new_error
                            best_adjustment = (factor, idx, c, delta)
            
            if best_adjustment is None:
                # Try adding a new column if rank allows
                if r < rank_schoolbook:
                    # This is complex, skip for now
                    break
                else:
                    break
            
            factor, idx, c, delta = best_adjustment
            factor[idx, c] += delta
        
        T = compute_tensor(U, V, W)
        error = np.sum((T - T_target)**2)
        return U, V, W, error < 1e-10
    
    # Try decreasing ranks
    for r in range(80, 96):
        if time.time() > deadline:
            break
        U_try, V_try, W_try, success = try_rank(r, max_iter=500)
        if success:
            best_rank = r
            best_U = U_try
            best_V = V_try
            best_W = W_try
            break
    
    # Convert to lists with integer/half-integer entries
    def to_list(matrix):
        result = []
        for row in matrix:
            result.append([int(x) if abs(x - round(x)) < 1e-10 else float(x) for x in row])
        return result
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
