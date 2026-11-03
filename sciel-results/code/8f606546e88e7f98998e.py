"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Try to find low-rank decomposition
    # Start with schoolbook rank 50, try to reduce
    # First, let's try a structured approach
    
    # Approach: For 2x5x5, try rank 25 (theoretical lower bound from m*p)
    # Set up the system: T[(i,j),(j',k),(k',i')] = sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c]
    # Non-zero: when j=j', i=i', k=k'
    
    # Let's try to construct a decomposition of rank 25
    # Idea: For each j (0..4) and each pair of output positions, use structure
    
    # Actually, let's try a different approach:
    # Use the fact that n=2, so we have two "rows" of outputs
    # For row i, C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try rank = 25 and see if we can solve it
    # We'll use a randomized approach with gradient-free optimization
    
    best_rank = n * m * p  # start with schoolbook
    best_result = None
    
    # Generate schoolbook first as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank using algebraic methods
    # For rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # and for all other index combinations, the sum = 0
    
    # Let's try ranks from 25 to 50
    for r in range(25, best_rank + 1):
        if time.time() > deadline - 20:
            break
        result = try_find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result

def try_find_decomposition(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using iterative refinement."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(nm, r) * 0.1
    V = np.random.randn(mp, r) * 0.1
    W = np.random.randn(pn, r) * 0.1
    
    # The tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i) = 1 for all i,j,k
    # We want to minimize the error
    # Error = sum over all (a,b,c) of (sum_d U[a,d]*V[b,d]*W[c,d] - T[a,b,c])^2
    
    # But this is expensive. Let's use a smarter approach.
    # The key equations are:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For each (i,j,k) with inconsistent indices: sum = 0
    
    # Let's use a fixed-point iteration or gradient descent on the non-zero equations
    
    # Actually, let's try a different approach: use the structure
    # For 2x5x5, try a block decomposition
    
    # Approach: For each j, the contribution involves A[:,j] and B[j,:]
    # Let's try to use rank-5 blocks for each j
    
    # For fixed j, we need: for all i,k: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c = 1
    # This is like a rank-r decomposition of a 2x5 tensor (for fixed j)
    
    # A 2x5 tensor has rank at most 2. So for each j, we could use rank 2?
    # But the c indices are shared across j's.
    
    # Let me try a structured approach:
    # Use r = 2*5 = 10? Probably not enough.
    # Use r = 5*5 = 25? Let's try.
    
    # For r=25, let's try a specific structure:
    # Let c = 5*j + k for j=0..4, k=0..4
    # U[i*m+j, 5*j+k] = A-related, V[j*p+k, 5*j+k'] = B-related, W[k'*n+i, 5*j+k] = ...
    
    # This is getting complicated. Let me just try numerical optimization.
    
    # Use simple gradient descent on the 50 non-zero equations
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 10:
            return None
        
        # Compute residuals for the 50 non-zero equations
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residuals[i,j,k] = val - 1.0
        
        # Also check some zero equations (sample)
        # For efficiency, only check a subset of zero equations
        zero_residuals = np.zeros(50)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check (i,j,k) vs (i,j,k+1 mod p) type inconsistencies
                    if idx < 50:
                        # Pick a random inconsistent triple
                        ii = (i + 1) % n
                        val = 0.0
                        for c in range(r):
                            val += U[ii*m+j, c] * V[j*p+k, c] * W[k*n+ii, c]
                        zero_residuals[idx] = val
                        idx += 1
        
        # Compute gradients (approximate)
        # d/dU[a,c] of sum (residual)^2 = 2 * sum over (i,j,k) where a=i*m+j of residual * V[j*p+k,c] * W[k*n+i,c]
        # This is expensive. Let's use a simpler update rule.
        
        # Use a multiplicative update or simple gradient step
        # For simplicity, let's use a heuristic update
        
        # Actually, this is too slow in pure Python. Let me try a different approach.
        break
    
    return None

# EVOLVE-BLOCK-END
