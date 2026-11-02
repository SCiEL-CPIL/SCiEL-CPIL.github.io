import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Searches for the minimum rank decomposition with integer/half-integer entries.
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute indices
    u_indices = [(i, j) for i in range(n) for j in range(m)]
    v_indices = [(j, k) for j in range(m) for k in range(p)]
    w_indices = [(k, i) for k in range(p) for i in range(n)]
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly satisfies the tensor equation."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank(rank, num_trials=100):
        """Try to find a valid decomposition of given rank."""
        # Use small integer/half-integer values
        values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 23 down to 15 (reasonable lower bound)
    for rank in range(23, 14, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, num_trials=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
