"""Matrix multiplication 4,4,7 - structured tensor decomposition search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe under 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Try structured U matrices (Kronecker-like) and solve for V, W
    # The constraint is bilinear in (V,W) for fixed U, so we can solve as linear system
    
    def check_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def solve_for_VW(U):
        """Given U, solve for V and W such that the decomposition holds."""
        # For each c, we have U[:,c], V[:,c], W[:,c]
        # Constraint: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear in (V,W) jointly. 
        # Alternative: fix V, solve for W (or vice versa)
        # Or: treat it as - for each c, the outer product structure
        # 
        # Actually, let's think of it differently.
        # Let A_c = U[:,c] (vector of length n*m)
        # Let B_c = V[:,c] (vector of length m*p)  
        # Let C_c = W[:,c] (vector of length p*n)
        # Constraint: sum_c A_c[i*m+j] * B_c[j*p+k] * C_c[k*n+i] = 1
        #
        # This is a 3-tensor decomposition problem.
        # For fixed U, this is still hard.
        #
        # Alternative approach: use the fact that matrix multiplication tensor
        # has a specific structure. Let's try to build it from smaller pieces.
        
        return None
    
    # Better approach: use known constructions
    # For 2x2x2: rank 7 (Strassen)
    # For 4x4x4: can use tensor product of 2x2x2 -> rank 49, or better constructions
    # Known: 4x4x4 multiplication rank is at most 49 (Strassen tensor product)
    # But better: rank 27 is achievable for 4x4x4
    
    # For 4x4x7: Let's try to split 7 = 4 + 3
    # 4x4x4 part: use rank-27 construction
    # 4x4x3 part: need to handle the remaining 3 columns
    
    # Actually, let me try a different structural approach.
    # The multiplication (4x4)*(4x7) -> (4x7)
    # We can view this as 7 separate 4x4x1 multiplications? No...
    # Each output column k depends on all input rows j of B and all rows i of A.
    
    # Let me try: use the 2x2 Strassen construction as a building block
    # 4x4 = [2x2 2x2; 2x2 2x2]
    # 4x7 = [2x4 2x3; 2x4 2x3] approximately
    
    # This is getting complex. Let me try a direct computational approach
    # with a structured U.
    
    # Approach: Try U as a block diagonal or Kronecker structure
    # and use iterative methods to find V, W
    
    # Simplest improvement: try to find any rank < 112 decomposition
    # by using the linearity of the problem
    
    # Key insight: The matrix multiplication tensor T of size (nm) x (mp) x (pn)
    # can be decomposed. The schoolbook uses nm*p = 112 rank-1 tensors.
    # We can try to combine them.
    
    # Let's try: use random structured U and solve the resulting system
    # For fixed U, the constraints on (V,W) are:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Let X_c = V[:,c] and Y_c = W[:,c]
    # This is: sum_c U_ij,c * X_c(jp+k) * Y_c(kn+i) = 1
    # 
    # If we fix all V, this becomes linear in W:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Let alpha_{i,j,k,c} = U[i*m+j,c] * V[j*p+k,c]
    # Then: sum_c alpha_{i,j,k,c} * W[k*n+i,c] = 1
    # This is a linear system in W!
    # 
    # Similarly, if we fix W, it's linear in V.
    # So we can use alternating minimization!
    
    def alternating_minimization(U, max_iter=200, tol=1e-8):
        """Given U, find V and W via alternating linear solves."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Initialize V and W randomly
        rng = np.random.RandomState(42)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        best_residual = float('inf')
        best_V, best_W = V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for W given U and V:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by (k,i): for fixed (k,i), the equation involves W[k*n+i,c] for all c
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for all j
            # So for fixed (k,i), we have m equations in rank unknowns W[k*n+i,0..rank-1]
            # A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
            # sum_c A_{j,c} * W[k*n+i,c] = 1 for j=0..m-1
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # Build system: m equations, rank unknowns
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # Solve A * W_vec = ones(m)
                    # Least squares
                    try:
                        W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = W_vec
                    except:
                        W_new[k * n + i, :] = 0.0
            W = W_new
            
            # Solve for V given U and W:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by (j,k): for fixed (j,k), the equation involves V[j*p+k,c] for all c
            # sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c] = 1 for all i
            # A_{i,c} = U[i*m+j,c] * W[k*n+i,c]
            # sum_c A_{i,c} * V[j*p+k,c] = 1 for i=0..n-1
            
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        V_vec, residuals, rank_sys, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = V_vec
                    except:
                        V_new[j * p + k, :] = 0.0
            V = V_new
            
            # Check residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual += (s - 1.0) ** 2
            residual = np.sqrt(residual / (n * m * p))
            
            if residual < best_residual:
                best_residual = residual
                best_V, best_W = V.copy(), W.copy()
            
            if residual < tol:
                break
        
        return best_V, best_W, best_residual
    
    # Try different structured U matrices
    # Strategy 1: Kronecker product structure
    # 4 = 2*2, so try U as Kronecker of 2x2 blocks
    
    # First, let's get a baseline: try to improve on schoolbook
    # The schoolbook has U as identity-like (each column has one 1)
    
    # Try structured U: block diagonal with 2x2 Strassen-like structure
    # For 2x2x2 Strassen: rank 7
    # U_strassen for 2x2x2:
    # This is complex. Let me try a simpler approach.
    
    # Try: U with rank r, where each column is a Kronecker product
    # of vectors from smaller spaces
    
    # Actually, let me try a completely different approach:
    # Use the known fact that matmul(4,4,4) has rank at most 49 (Strassen)
    # and try to extend to (4,4,7)
    
    # For now, let's try to find ANY decomposition with rank < 112
    # using the alternating minimization with a good U
    
    # Try U as random with different ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Start with schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    if check_decomposition(U_school, V_school, W_school):
        best_result = (U_school, V_school, W_school, best_rank)
    
    # Try to improve: use alternating minimization with structured U
    # Try ranks from 100 down to see if we can find a solution
    
    rng = np.random.RandomState(123)
    
    # Try a few structured U patterns
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        for rank in [100, 95, 90, 85, 80]:
            if time.time() > deadline - 15:
                break
            
            # Create structured U: use Kronecker-like structure
            # Divide 4x4 into 2x2 blocks
            U = np.zeros((n * m, rank))
            
            # Fill U with structured pattern
            # Use 2x2 block structure
            for block_i in range(2):
                for block_j in range(2):
                    # Each 2x2 block gets rank/4 columns
                    block_rank = rank // 4
                    start = (block_i * 2 + block_j) * block_rank
                    end = start + block_rank
                    for c in range(start, end):
                        for ii in range(2):
                            for jj in range(2):
                                i = block_i * 2 + ii
                                j = block_j * 2 + jj
                                U[i*m+j, c] = rng.randn()
            
            V, W, residual = alternating_minimization(U, max_iter=100)
            
            if residual < 1e-6:
                # Found a valid decomposition!
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W, rank)
                    break
            elif rank < best_rank and residual < 1e-4:
                # Close but not exact - keep as candidate
                pass
    
    # If we found an improvement, use it; otherwise use schoolbook
    if best_result is not None:
        U, V, W, rank = best_result
    else:
        U, V, W, rank = U_school, V_school, W_school, n*m*p
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

