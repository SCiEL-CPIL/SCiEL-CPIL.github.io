"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition using structured search
    # Strategy: Use the fact that we can exploit linear structure in the middle dimension
    
    # Approach: Try to use a "block" decomposition where we group the middle index
    # For m=5, we can try to represent the sum over j=0..4 using fewer terms
    
    # Let's try a local search starting from a slightly perturbed schoolbook
    np.random.seed(42)
    
    # Try ranks from 104 down
    for target_rank in range(rank_schoolbook - 1, 80, -1):
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook-like structure but with some merging
        # Try to merge pairs of terms that have overlapping support
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # First, try a greedy merging approach
        # Start with all 105 terms, try to merge them
        # This is complex; instead, let's try random initialization with structure
        
        # Random search with integer/half-integer entries
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U_r = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V_r = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            W_r = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
            
            # Scale to make it more likely to work
            # The product U*V*W should sum to 1 for each (i,j,k)
            # Average term magnitude should be ~ 1/(target_rank * typical_product)
            
            # Check if this works (unlikely for random, but try)
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                break
        
        if best_rank < rank_schoolbook:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
