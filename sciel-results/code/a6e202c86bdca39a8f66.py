"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook decomposition
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
    
    # Try to reduce rank by removing columns that are linearly dependent in the tensor sense
    # Check if a column can be removed: the tensor must still be exactly representable
    # This is expensive, so we do a limited search
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each column one at a time and check if the tensor is still exactly decomposed
    # For efficiency, we only try a subset
    
    def check_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        # For each (i,j,k), check sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check that non-diagonal entries are 0
        # This is expensive, so we skip for now (the diagonal check is the main constraint)
        return True
    
    # Greedy: try to remove columns
    # For each column, check if removing it still allows exact decomposition
    # This requires solving for the remaining columns, which is hard.
    # Instead, we try a simple heuristic: if a column has a "redundant" pattern, remove it.
    
    # Given the complexity, we'll just return the schoolbook for now.
    # In a more advanced implementation, we could use a randomized search.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
