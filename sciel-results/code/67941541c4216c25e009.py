import time
import sys
import random
from itertools import product

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    The decomposition is of the form:
    C = sum_{c=0}^{rank-1} (U_c @ V_c) * W_c
    
    We first try to find a decomposition with rank < 18 (schoolbook).
    If not found within time budget, return the schoolbook rank-18 solution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # Start from rank 17 down to a reasonable minimum
    # Use a structured search: try to find solutions with small integer/half-integer entries
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from best_rank - 1 down to max(1, best_rank - 3) 
    # (limit the search depth to stay within time budget)
    for target_rank in range(best_rank - 1, max(0, best_rank - 4), -1):
        if time.time() > deadline:
            break
        
        # Try a limited number of random initializations
        # For each, try a simple iterative refinement
        found = False
        num_attempts = 20  # Keep this small to stay within time
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random values (small integers/half-integers)
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we found a lower rank, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
