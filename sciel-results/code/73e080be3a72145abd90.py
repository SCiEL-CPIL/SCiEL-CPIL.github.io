"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a lower-rank decomposition for 5x5x5 matrix multiplication.
    Falls back to schoolbook (rank 125) if no better solution found within time budget.
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    schoolbook_rank = n * m * p  # 125
    
    # Build schoolbook solution as fallback
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
    
    best_solution = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition using random search
    # We'll use a simple approach: try random matrices and check if they work
    # This is a heuristic approach that may not always find the optimal solution
    
    # The matrix multiplication tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # We want to find rank-r decomposition:
    # C[i,k] = sum_c U[i,c] * V[c,k]  (but this is not quite right for matmul)
    
    # Actually, the tensor decomposition for matmul is:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    # where U, V, W are the factor matrices
    
    # Let's try a random search with small ranks
    # This is a very simplified approach; a proper search would be much more complex
    
    # For now, just return the schoolbook solution since finding a better one
    # in the time budget is non-trivial without a proper optimization framework
    
    return best_solution
# EVOLVE-BLOCK-END
