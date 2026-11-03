"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Build target tensor T[i,j,k] = 1 if we multiply (i,j) * (j,k) -> (i,k)
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # We can compute this more efficiently using matrix operations
    
    # Build the full tensor from U, V, W
    # T_decomp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Efficient approach: for each c, compute outer product contributions
    # But for small sizes, direct loop is fine
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    
    return max_err < 1e-9

def _try_rank_search(n, m, p, max_rank, time_limit=140.0):
    """Try to find a decomposition with rank <= max_rank using random search."""
    start_time = time.time()
    deadline = start_time + time_limit
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from lower to higher
    for target_rank in range(1, max_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        num_tries = 100
        for trial in range(num_tries):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize U, V, W with random values
            # Use structured initialization inspired by algebraic geometry
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Add some structure: make each row have a dominant component
            for r in range(target_rank):
                U[np.random.randint(n*m), r] += 1.0
                V[np.random.randint(m*p), r] += 1.0
                W[np.random.randint(p*n), r] += 1.0
            
            # Verify
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                        err = max(err, abs(s - 1.0))
            
            if err < 1e-9:
                # Found a valid decomposition!
                # Convert to lists with integer/half-integer entries if possible
                U_list = [[float(x) for x in U[row]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row]] for row in range(p*n)]
                
                # Try to round to half-integers
                U_rounded = [[round(2*x)/2 for x in row] for row in U_list]
                V_rounded = [[round(2*x)/2 for x in row] for row in V_list]
                W_rounded = [[round(2*x)/2 for x in row] for row in W_list]
                
                # Verify rounded version
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = target_rank
                    return best_result
                else:
                    # Use unrounded if it works
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best_result
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # First, get the schoolbook decomposition as a safe fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    # Known bounds: rank(3x5x5) is at most 75, conjectured to be lower
    # Try searching for ranks significantly below schoolbook
    search_max_rank = min(schoolbook_rank - 1, 60)
    
    result = _try_rank_search(n, m, p, search_max_rank, time_limit=140.0)
    
    if result is not None:
        return result
    else:
        return schoolbook
# EVOLVE-BLOCK-END
