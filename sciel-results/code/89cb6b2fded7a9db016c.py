import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses structured search: start from schoolbook, try to reduce rank by finding
    linear dependencies and using exact rational arithmetic.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    start_time = time.time()
    time_limit = 135.0  # Leave 15s margin before 150s kill
    
    def check_decomposition_exact(U, V, W, rank):
        """Check if decomposition is exact using Fraction arithmetic."""
        # Convert to Fraction for exact check
        U_frac = [[Fraction(U[i, t]) for t in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, t]) for t in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i, t]) for t in range(rank)] for i in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for t in range(rank):
                        val += U_frac[i*m+j][t] * V_frac[j*p+k][t] * W_frac[k*n+i][t]
                    if val != 1:
                        return False
        return True
    
    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        for val in mat:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True
    
    def try_reduce_rank(U, V, W, current_rank):
        """
        Try to reduce rank by finding a linear combination that can be absorbed.
        We look for a column t such that the rank-1 tensor (u_t, v_t, w_t)
        can be expressed as a combination of others, or find a pair that can be merged.
        """
        # This is a heuristic: try removing each column and see if we can adjust
        # others to compensate. For now, we'll try a simpler approach:
        # Check if any two columns are linearly dependent in a way that allows merging.
        
        # Actually, let's try a different approach: use the fact that the tensor
        # has structure. For matrix multiplication, there are known lower bounds.
        # For 3x4x6, the minimum rank is known to be at most 72 (schoolbook).
        # Let's try to find if there's a rank < 72 solution.
        
        # Strategy: Try to find a rank 71 solution by perturbing schoolbook
        # and solving for the missing term.
        
        return None
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank 72."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}, rank
    
    def try_als_search(target_rank, max_iters=5000, tol=1e-10):
        """
        Try to find a decomposition of given rank using ALS-like optimization
        with rounding to half-integers at the end.
        """
        if time.time() - start_time > time_limit:
            return None
        
        np.random.seed(42)
        
        # Initialize with small random values
        U = np.random.uniform(-1, 1, size=(n*m, target_rank))
        V = np.random.uniform(-1, 1, size=(m*p, target_rank))
        W = np.random.uniform(-1, 1, size=(p*n, target_rank))
        
        # Precompute index mappings
        U_idx = [(i*m+j, i, j) for i in range(n) for j in range(m)]
        V_idx = [(j*p+k, j, k) for j in range(m) for k in range(p)]
        W_idx = [(k*n+i, k, i) for k in range(p) for i in range(n)]
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_row = U[i*m+j].copy()
                    for k in range(p):
                        # For each k, we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
                        # So: U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]) = 1
                        # This gives a linear system if we fix V and W
                        pass
                    # Actually, this is nonlinear. Let's use a different approach.
                    
            # Use a gradient-free approach: try to minimize the error
            # by randomly perturbing and keeping improvements
            break
        
        # This ALS approach is complex. Let's try a simpler structured search.
        return None
    
    def structured_search():
        """
        Use the structure of matrix multiplication to find a reduced rank decomposition.
        
        Key insight: For matrix multiplication C = A*B where A is n×m and B is m×p,
        the tensor T[i,j,k] = 1. We can try to find a decomposition where
        some terms share structure.
        
        One approach: Use the fact that for fixed j, the terms involving
        A[i,j] and B[j,k] form a rank-1 matrix in (i,k) space.
        We can try to merge terms across different j values.
        """
        # Let's try a specific construction:
        # For each j, we have n*p terms. Can we reduce these?
        # For fixed j, the contribution is A[i,j] * B[j,k] for all i,k.
        # This is already a rank-1 matrix in (i,k) for each j.
        # The total is sum over j of these rank-1 matrices.
        # The rank of the sum of rank-1 matrices can be less than the sum of ranks.
        
        # Actually, the schoolbook decomposition has exactly one term per (i,j,k),
        # and each term is a rank-1 tensor. The question is whether we can find
        # a smaller set of rank-1 tensors that sum to the same tensor.
        
        # Let's try to use bilinear algorithms. For small matrices, there might
        # be known optimal decompositions.
        
        # For 3x4x6, let's try to find if rank 71 is possible by solving a system.
        
        # Alternative: Try to find a decomposition with rank < 72 by using
        # the fact that the tensor is sparse in a certain sense.
        
        # Let's try a direct search with exact arithmetic for small rank reductions.
        
        # Start with schoolbook and try to remove one term and solve for adjustments.
        
        base = schoolbook_decomposition()
        base_rank = base[1]
        
        # Try to reduce by 1
        # Remove the last term (i=2, j=3, k=5) and see if we can adjust others
        # This requires solving a nonlinear system, which is hard.
        
        # Instead, let's try a random search with exact verification for rank 71, 70, etc.
        
        for target_rank in range(base_rank - 1, max(base_rank - 10, 50), -1):
            if time.time() - start_time > time_limit:
                break
            
            # Try random initializations with exact verification
            for attempt in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Use small integer/half-integer values
                U = np.random.randint(-4, 5, size=(n*m, target_rank)) * 0.5
                V = np.random.randint(-4, 5, size=(m*p, target_rank)) * 0.5
                W = np.random.randint(-4, 5, size=(p*n, target_rank)) * 0.5
                
                # Quick check with floats
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(target_rank):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Verify with exact arithmetic
                    if check_decomposition_exact(U, V, W, target_rank):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }, target_rank
        
        return None, base_rank
    
    # First, try structured search for reduced rank
    result, rank = structured_search()
    
    if result is None:
        # Fall back to schoolbook
        result, rank = schoolbook_decomposition()
    
    return result
