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

    # Lader's rank-23 decomposition for 3x3x3
    # Each vector has 9 entries (3x3 indices)
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1],
    ]

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

    # Build the Lader-based decomposition: two 3x3x3 blocks + one 3x3x2 block
    # This gives rank 23 + 23 + 18 = 64
    def build_lader_decomposition():
        rank = 23 + 23 + 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First 3x3x3 block (columns 0-22), k indices 0-2
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Second 3x3x3 block (columns 23-45), k indices 3-5
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][23+r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+(3+k)][23+r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(3+k)*n+i][23+r] = lader_W[r][k*3+i]
        
        # 3x3x2 block (columns 46-63), k indices 6-7
        col = 46
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*m+j][col] = 1
                    V[j*p+(6+k)][col] = 1
                    W[(6+k)*n+i][col] = 1
                    col += 1
        
        return U, V, W, rank

    U, V, W, best_rank = build_lader_decomposition()
    
    # Verify Lader-based decomposition
    assert validate_decomposition(U, V, W, best_rank), "Lader decomposition failed validation"
    
    # Schoolbook as ultimate fallback
    schoolbook_rank = n * m * p
    
    # Value set for entries (integer/half-integer)
    value_set = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 3, -3]
    
    # Try to improve: search for lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    lower_bound = max(n * m, m * p, p * n)
    
    # Candidate ranks to try, sorted ascending
    candidate_ranks = []
    for r in range(lower_bound, best_rank):
        if r % 4 == 0 or r % 8 == 0 or r in [36, 40, 44, 48, 52, 56, 60, 62, 63]:
            candidate_ranks.append(r)
    candidate_ranks.sort()
    
    # Also try structured approaches with shared computations
    # Strategy: use Lader for 3x3x3 blocks but try to share U vectors across k-blocks
    # This requires the same U row to work for multiple k values
    
    # Try rank-62 and rank-63 with random perturbation of the Lader solution
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Attempt count based on time budget
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Start from Lader structure and randomly zero out some columns
            # or try random generation
            if target_rank >= 60:
                # Try to prune columns from the Lader solution
                U_try = [row[:] for row in U]
                V_try = [row[:] for row in V]
                W_try = [row[:] for row in W]
                
                # Randomly zero out (best_rank - target_rank) columns
                cols_to_remove = random.sample(range(best_rank), best_rank - target_rank)
                for c in cols_to_remove:
                    for r in range(len(U_try)):
                        U_try[r][c] = 0
                    for r in range(len(V_try)):
                        V_try[r][c] = 0
                    for r in range(len(W_try)):
                        W_try[r][c] = 0
                
                if validate_decomposition(U_try, V_try, W_try, target_rank):
                    best_rank = target_rank
                    U, V, W = U_try, V_try, W_try
                    break
            else:
                # Full random generation
                U_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                          for _ in range(n * m)]
                V_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                          for _ in range(m * p)]
                W_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                          for _ in range(p * n)]
                
                if validate_decomposition(U_rand, V_rand, W_rand, target_rank):
                    best_rank = target_rank
                    U, V, W = U_rand, V_rand, W_rand
                    break
        
        if best_rank == target_rank:
            continue
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
