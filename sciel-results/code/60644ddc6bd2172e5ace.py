"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to decompose T as sum of rank-1 tensors
    # Each rank-1 term has the form u_c * v_c * w_c where:
    #   u_c is a vector of size n*m (indexed by (i,j))
    #   v_c is a vector of size m*p (indexed by (j,k))
    #   w_c is a vector of size p*n (indexed by (k,i))
    # and T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Strategy: Start with schoolbook rank (n*m*p = 72) and try to find lower rank
    # by formulating as a binary optimization problem.
    
    # First, let's generate candidate rank-1 terms from various algebraic patterns.
    # For 3x3*3x8, we can try:
    # 1. Schoolbook terms
    # 2. Strassen-like combinations
    # 3. Random sparse patterns
    
    # Let's use a different approach: we'll try to find a low-rank decomposition
    # by using the fact that the tensor has a lot of structure.
    
    # For 3x3 * 3x8, the minimum rank is unknown but likely less than 72.
    # Let's try a systematic search with increasing rank.
    
    # Actually, let's think about this differently. The tensor T is defined by:
    # T[i, j, k] = 1 for all valid indices
    # This is a "all-ones" tensor in the appropriate dimensions.
    
    # The key insight: for matrix multiplication A (n x m) * B (m x p) = C (n x p),
    # the tensor has T[i,j,k] = 1 where i indexes rows of A/cols of C, 
    # j indexes cols of A/rows of B, k indexes cols of B/rows of C.
    
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{c=0}^{r-1} (sum_{j} A[i,j] * U_c[i,j]) * V_c[j,k] * W_c[k,i]
    # Wait, that's not quite right. Let me reconsider.
    
    # Standard formulation:
    # C[i,k] = sum_c U_c[i] * V_c[k] where
    # U_c[i] = sum_j A[i,j] * u_c[j]  (linear in A)
    # V_c[k] = sum_j B[j,k] * v_c[j]  (linear in B)
    
    # So the decomposition is:
    # C[i,k] = sum_c (sum_j u_c[j] * A[i,j]) * (sum_l v_c[l] * B[j,l])
    #         = sum_j,l A[i,j] * B[j,l] * (sum_c u_c[j] * v_c[l])
    
    # For this to equal the standard matrix product sum_j A[i,j] * B[j,l] (with l=k),
    # we need: sum_c u_c[j] * v_c[l] = delta(j, l) for all j, l
    
    # So the condition is: U * V^T = I_m where U is (r x m) and V is (r x m)
    # Wait, that's for square matrices. Let me be more careful.
    
    # For A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-r decomposition:
    # C[i,k] = sum_c (sum_j A[i,j] * u_c[i,j]) * (sum_j B[j,k] * v_c[j,k])
    
    # For this to work for all A, B:
    # sum_c u_c[i,j] * v_c[j',k] = delta(i,i') * delta(j,j') * delta(k,k')... 
    # No, that's not right either.
    
    # Let me use the standard tensor formulation. The multiplication tensor M has:
    # M[i, j, k] = 1 if we're computing C[i,k] = A[i,j] * B[j,k]
    
    # A rank-r decomposition is:
    # M = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector indexed by (i,j) pairs, V_c by (j,k) pairs, W_c by (i,k) pairs
    
    # Specifically:
    # M[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[i,k]
    
    # For the standard multiplication, M[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] = 1 for all i,j,k
    
    # This is a system of equations in the unknowns U_c, V_c, W_c.
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can try known decompositions or use random restarts with local search.
    
    # For 3x3 * 3x8, let's try to find a decomposition of rank < 72.
    
    # Known results: 3x3 matrix multiplication has rank at most 23 (Landsberg, 1997)
    # and at least 19. For 3x3 * 3x8, the rank should be somewhere between these.
    
    # Let's try a practical approach: use the schoolbook decomposition and then
    # try to merge terms using linear algebra.
    
    # Actually, let's just try to find any valid decomposition with rank less than 72.
    # We'll use a randomized approach with local optimization.
    
    # First, let's verify that our understanding is correct by implementing a
    # verification function.
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is correct."""
        r = len(U[0]) if U else 0
        # Check dimensions
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for c in range(r):
            if len(U[c]) != 1 or len(V[c]) != 1 or len(W[c]) != 1:
                # U, V, W are lists of lists, each inner list has one element (the coefficient)
                # Actually, U[i][c] is the coefficient
                pass
        
        # Reinterpret: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
        # M[i,j,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        
        # Check that M[i,j,k] = 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, rank = schoolbook_decomposition(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies
    # This is complex, so let's try a simpler approach:
    # Try random sparse decompositions with decreasing rank
    
    # Actually, for this specific problem, let's try a known technique:
    # Use the fact that 3x3 matrix multiplication can be done with rank 23,
    # and 3x3 * 3x8 should be doable with rank proportional to that.
    
    # Let's try a systematic search: for rank from 50 down to 30, try to find
    # a valid decomposition using random initialization and local search.
    
    # But this might be too slow. Let's just return the schoolbook decomposition
    # if we can't find something better quickly.
    
    # Try a few random attempts
    np.random.seed(42)
    
    for trial_rank in [64, 56, 48, 40, 32]:
        if time.time() > deadline - 10:
            break
        if trial_rank >= best_rank:
            continue
            
        # Try random sparse initialization
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            U_try = np.zeros((n * m, trial_rank))
            V_try = np.zeros((m * p, trial_rank))
            W_try = np.zeros((p * n, trial_rank))
            
            # Initialize with random sparse patterns
            for c in range(trial_rank):
                # Each term has a few non-zero entries
                nnz_u = np.random.randint(1, 4)
                nnz_v = np.random.randint(1, 4)
                nnz_w = np.random.randint(1, 4)
                
                idx_u = np.random.choice(n * m, nnz_u, replace=False)
                idx_v = np.random.choice(m * p, nnz_v, replace=False)
                idx_w = np.random.choice(p * n, nnz_w, replace=False)
                
                for idx in idx_u:
                    U_try[idx, c] = np.random.choice([-1, 1])
                for idx in idx_v:
                    V_try[idx, c] = np.random.choice([-1, 1])
                for idx in idx_w:
                    W_try[idx, c] = np.random.choice([-1, 1])
            
            # Convert to list format
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_try.tolist()
            
            if verify(U_list, V_list, W_list, n, m, p):
                best_rank = trial_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    # Check deadline
    if time.time() > deadline - 2:
        # Return best found
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If we still have the schoolbook, return that
    if best_rank == n * m * p:
        U, V, W, _ = schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
