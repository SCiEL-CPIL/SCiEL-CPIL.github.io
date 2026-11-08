"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

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
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact.
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        We need: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(target_rank, seed=42):
        """Try to find a decomposition of given rank using randomized search.
        Returns (U, V, W) if found, else None.
        """
        np.random.seed(seed)
        
        # Strategy: Start from schoolbook and try to merge terms
        # Or use a more direct randomized approach
        
        # Approach 1: Try to find a structured decomposition
        # For small matrices, we can try various patterns
        
        # Let's try a greedy approach: start with all-zero matrices and fill in
        # rank-1 components that cover the most uncovered (i,j,k) triples
        
        # First, let's try a simple randomized initialization and see if we can
        # iteratively improve
        
        # Actually, let's try a different approach: use the fact that for
        # specific small dimensions, there might be known lower-rank decompositions.
        
        # For 3x5x7, the schoolbook is 105. Let's try to reduce.
        
        # Method: Randomized Kronecker substitution / greedy peeling
        # We represent the tensor as a sum of rank-1 terms.
        
        # Let's try: initialize U, V, W randomly with small integers, then
        # use gradient-like updates or local search to satisfy the constraints.
        
        # Simpler: Try specific structured patterns.
        
        # Pattern 1: Try to use fewer terms by sharing structure
        # For example, if we can find that some (i,j,k) terms can be combined.
        
        # Let's try a direct search for small rank reductions.
        # Start by trying to see if rank 104 is possible, then 103, etc.
        
        # But this is hard. Let's try a randomized approach:
        
        # Initialize U, V, W with random values from {-1, 0, 1}
        # Then check if we can satisfy the constraints.
        
        # This is unlikely to work directly, but let's try a few seeds.
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            # Check if this works
            if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try ranks from just below schoolbook down to a reasonable lower bound
    # Lower bound: max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    # But realistically, we might not get much below 105 for 3x5x7
    
    # Try a few ranks near schoolbook to see if we can shave off a few terms
    for target_rank in range(rank_schoolbook - 1, max(35, rank_schoolbook - 20), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, seed=target_rank)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
