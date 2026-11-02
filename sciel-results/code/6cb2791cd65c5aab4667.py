"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Validate function using numpy for speed
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook works
    assert validate_decomposition(U, V, W, rank), "Schoolbook decomposition failed validation"
    
    # Strategy: Try to find lower-rank decompositions using structured patterns
    # Key insight: For 3x3*3x3, Laderman gives rank 20. 
    # For 3x3*3x8, we can potentially use shared computations.
    
    # Try a structured approach: split 8 columns into groups and use shared subcomputations
    # Value set for entries (integer/half-integer)
    value_set = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 3, -3]
    
    # Try ranks from lower bound up to best_rank - 1
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # Focus on ranks that are plausible: multiples of 4 or 8 near the lower bound
    candidate_ranks = []
    for r in range(lower_bound, best_rank):
        if r % 4 == 0 or r % 8 == 0 or r in [36, 40, 44, 48, 52, 56, 60, 64, 68]:
            candidate_ranks.append(r)
    
    # Sort by rank (try lower ranks first)
    candidate_ranks.sort()
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random attempts at this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random matrices with constrained values
            U_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(n * m)]
            V_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(m * p)]
            W_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(p * n)]
            
            if validate_decomposition(U_rand, V_rand, W_rand, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
        
        # If we found a solution at this rank, try to go lower
        if best_rank == target_rank:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
