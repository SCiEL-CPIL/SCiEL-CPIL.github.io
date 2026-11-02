import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n + i, r] follows trace convention
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_factors = None
    
    # Create schoolbook solution as fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_factors = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from low to high, use random init + iterative refinement
    
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But actually the tensor rank is at least the max of the "slice" ranks
    
    # Let's try a systematic approach: for each candidate rank, try to solve
    # for factors that satisfy the tensor equation
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        # Compute T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Check against all-ones tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 10.0:
            return None
        
        # Use random initialization and try to solve
        # The equation is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Strategy: fix U and V randomly, solve for W, then check
        # Or use alternating minimization
        
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization: fix two, solve for third
            # This is tricky because it's a 3-way tensor decomposition
            
            # Alternative: use the fact that for fixed i,j, we have:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system in W if U and V are fixed
            
            # Let's try a different approach: use structured initialization
            # based on the block structure
            
            # Initialize with a pattern that respects the j-index
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each j, allocate some rank budget
            # Try different allocations
            if rank >= m:
                per_j = rank // m
                rem = rank % m
                for j in range(m):
                    start = j * per_j + min(j, rem)
                    end = start + per_j + (1 if j < rem else 0)
                    for i in range(n):
                        for k in range(p):
                            if start < rank and end > start:
                                r = start
                                U[i*m+j, r] = 1.0
                                V[j*p+k, r] = 1.0
                                W[k*n+i, r] = 1.0
                                start += 1
                                if start >= end:
                                    break
            
            # Verify
            if verify(U, V, W):
                return (U, V, W)
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) / something, but let's be conservative
    # Actually, the rank must be at least the max over all "fibers"
    # For fixed i,j: the fiber over k has rank at most rank, and the vector is all-1s (rank 1)
    # For fixed i,k: the fiber over j... 
    
    # Let's just try from a reasonable lower bound
    # Theoretical lower bound for M_{n,m,p} tensor rank is at least max(n*m, m*p, p*n) / min(n,m,p)
    # But let's just try various ranks
    
    lower_bound = max(n*m, m*p, p*n)  # = 24
    # Actually, a better lower bound: the rank is at least the max of the
    # "slice" dimensions in a certain sense
    
    # Try from lower_bound to schoolbook rank
    for rank in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline - 15.0:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_factors = (U, V, W)
            # Check if we can go lower
            if rank == lower_bound:
                break
    
    # Convert to lists for return
    U, V, W = best_factors
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
