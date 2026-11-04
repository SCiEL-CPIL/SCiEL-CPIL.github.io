import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank decomposition
    # We search for rank R from best_rank - 1 down to max(1, best_rank - 5)
    # This is a heuristic; full search is infeasible.
    
    # Construct the target tensor T (flattened)
    # T[i, j, k] = 1
    # The bilinear form is sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
    # This is equivalent to matching the tensor T where T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # In the decomposition, the coefficient of A[i, j] B[j, k] in C[i', k'] is 
    # sum_r W[k' n + i', r] U[i m + j, r] V[j p + k, r]
    # We want this to be 1 if i'=i and k'=k, and 0 otherwise.
    
    # This is a system of equations.
    # Let's try rank 53.
    
    for R in range(best_rank - 1, max(1, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Use a simple local search (gradient descent on the error)
        # Error = sum_{i, j, k, i', k'} (Coeff - Target)^2
        # This is expensive to compute directly.
        # Instead, we can use the fact that the tensor is sparse.
        
        # For now, just return schoolbook if we don't have a specific algorithm.
        pass
    
    # Convert to list of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
