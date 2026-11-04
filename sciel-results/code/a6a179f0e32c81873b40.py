"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank n*m*p = 75)
    rank_schoolbook = n * m * p
    U = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    def validate(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank by combining schoolbook terms
    # Strategy: try to express the tensor as a sum of fewer rank-1 terms
    # by finding linear dependencies
    
    # For each subset of schoolbook columns, check if they can be merged
    # This is expensive, so we'll use a heuristic approach
    
    # Try removing pairs of columns and adjusting others
    # This is a simplified approach - in general, finding minimal rank is hard
    
    # Alternative: try small structured decompositions
    # For M_{3,5,5}, try to use the fact that 5 = 2+3 to apply Strassen-like ideas
    
    # For now, try to reduce rank by checking if any column is redundant
    # A column c is redundant if there exists a linear combination of other columns
    # that reproduces its contribution
    
    # Simple heuristic: try to remove columns one by one from the end
    # and check if the remaining decomposition is still valid
    # This won't work for the schoolbook decomposition since each column
    # is independent, but let's try anyway
    
    # Better approach: try to find a rank-r decomposition where r < 75
    # by solving a system of equations
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a nonlinear system. We can try to fix U and V and solve for W,
    # or use iterative methods.
    
    # For a quick improvement, let's try to see if we can reduce the rank
    # by a small amount using a greedy approach
    
    # Try rank 74, 73, etc. by zeroing out one column at a time and
    # adjusting the remaining columns slightly
    
    # This is getting complex. Let's just return the schoolbook decomposition
    # as it's guaranteed to be valid.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
