"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition
    # Strategy: try random perturbations and projections
    random.seed(42)
    
    # Try a few candidate ranks
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition with this rank
        # Use a simple approach: start from schoolbook and try to merge terms
        # This is a heuristic, not guaranteed to work
        
        # For now, just check if we can find any improvement
        # This is a placeholder for a more sophisticated search
        pass
    
    # Convert to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
