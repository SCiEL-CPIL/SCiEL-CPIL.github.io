"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of columns, check if we can merge them
    # A rank reduction is possible if there exists a vector c such that
    # the tensor equation still holds when we replace two rank-1 terms with one
    
    # Strategy: Try to find a linear combination of columns that preserves the tensor
    # We'll use a greedy approach: try to remove one column at a time by finding
    # if its contribution can be absorbed by others
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # and check it equals delta(i,j,k) for valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The tensor has 1s at all (i,j,k) positions
                    actual = 0.0
                    for r in range(r):
                        actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    # Try rank reductions by attempting to find linear dependencies
    # We'll try to find a non-trivial solution to the system where
    # we perturb the decomposition and see if we can reduce rank
    
    # For now, return the schoolbook decomposition
    # Converting to list format as required
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
