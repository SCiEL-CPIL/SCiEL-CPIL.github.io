"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook as fallback (rank 72)
    rank_school = n * m * p  # 72
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Strategy: Decompose 3x3x8 into two 3x3x4 problems.
    # The tensor for 3x3x4 has known rank 20.
    # We search for a rank-20 decomposition for 3x3x4.
    # If found, we duplicate it for the two halves of k (0-3 and 4-7).
    
    n_sub, m_sub, p_sub = 3, 3, 4
    
    # Precompute indices for 3x3x4 sub-problem
    # U indices: i*m + j (0..8)
    # V indices: j*p + k (0..11)
    # W indices: k*n + i (0..11)
    # Condition: sum_r U[i*3+j, r] * V[j*4+k, r] * W[k*3+i, r] == 1
    
    def check_sub_decomposition(U, V, W, rank):
        """Check if the 3x3x4 decomposition is exact."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s = 0
                    for r in range(rank):
                        s += U[i * m_sub + j][r] * V[j * p_sub + k][r] * W[k * n_sub + i][r]
                    if s != 1:
                        return False
        return True

    def compute_error_sub(U, V, W, rank):
        """Compute sum of squared errors for 3x3x4."""
        err = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m_sub + j][r] * V[j * p_sub + k][r] * W[k * n_sub + i][r]
                    err += (s - 1.0) ** 2
        return err

    def find_sub_decomposition(target_rank, time_limit):
        """Search for a rank-target_rank decomposition of 3x3x4."""
        start = time.time()
        
        # Initialize with random values
        U = [[0] * target_rank for _ in range(n_sub * m_sub)]
        V = [[0] * target_rank for _ in range(m_sub * p_sub)]
        W = [[0] * target_rank for _ in range(p_sub * n_sub)]
        
        for r in range(target_rank):
            for i in range(n_sub * m_sub):
                U[i][r] = random.uniform(-1, 1)
            for j in range(m_sub * p_sub):
                V[j][r] = random.uniform(-1, 1)
            for k in range(p_sub * n_sub):
                W[k][r] = random.uniform(-1, 1)
        
        best_err = float('inf')
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        # Local search
        while time.time() < start + time_limit:
            err = compute_error_sub(U, V, W, target_rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                if err < 1e-6:
                    # Try to round and verify
                    U_int = [[int(round(x)) for x in row] for row in U]
                    V_int = [[int(round(x)) for x in row] for row in V]
                    W_int = [[int(round(x)) for x in row] for row in W]
                    if check_sub_decomposition(U_int, V_int, W_int, target_rank):
                        return U_int, V_int, W_int
            
            # Perturb a random element
            mat = random.choice([0, 1, 2])
            if mat == 0:
                row = random.randint(0, n_sub * m_sub - 1)
                col = random.randint(0, target_rank - 1)
                U[row][col] += random.uniform(-0.5, 0.5)
            elif mat == 1:
                row = random.randint(0, m_sub * p_sub - 1)
                col = random.randint(0, target_rank - 1)
                V[row][col] += random.uniform(-0.5, 0.5)
            else:
                row = random.randint(0, p_sub * n_sub - 1)
                col = random.randint(0, target_rank - 1)
                W[row][col] += random.uniform(-0.5, 0.5)
        
        # Final check on best found
        U_int = [[int(round(x)) for x in row] for row in best_U]
        V_int = [[int(round(x)) for x in row] for row in best_V]
        W_int = [[int(round(x)) for x in row] for row in best_W]
        if check_sub_decomposition(U_int, V_int, W_int, target_rank):
            return U_int, V_int, W_int
        return None

    # Try to find a sub-decomposition of rank 20
    # If not, try 21, 22, etc.
    sub_found = None
    for target_sub_rank in [20, 21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
        
        # Give each attempt a portion of the remaining time
        time_per_attempt = min(15, max(5, (deadline - 10 - time.time()) / 2))
        sub_found = find_sub_decomposition(target_sub_rank, time_per_attempt)
        if sub_found:
            break
    
    if sub_found:
        U_sub, V_sub, W_sub = sub_found
        sub_rank = len(U_sub[0])
        
        # Construct the full 3x3x8 decomposition
        full_rank = 2 * sub_rank
        U_full = [[0] * full_rank for _ in range(n * m)]
        V_full = [[0] * full_rank for _ in range(m * p)]
        W_full = [[0] * full_rank for _ in range(p * n)]
        
        # First half: k in 0..3
        for r in range(sub_rank):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][r] = U_sub[i * m_sub + j][r]
            for j in range(m):
                for k in range(4):
                    V_full[j * p + k][r] = V_sub[j * p_sub + k][r]
            for k in range(4):
                for i in range(n):
                    W_full[k * n + i][r] = W_sub[k * n_sub + i][r]
        
        # Second half: k in 4..7
        offset = sub_rank
        for r in range(sub_rank):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][offset + r] = U_sub[i * m_sub + j][r]
            for j in range(m):
                for k in range(4):
                    V_full[j * p + (4 + k)][offset + r] = V_sub[j * p_sub + k][r]
            for k in range(4):
                for i in range(n):
                    W_full[(4 + k) * n + i][offset + r] = W_sub[k * n_sub + i][r]
        
        # Verify full decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(full_rank):
                        s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = full_rank
            best_result = {"U": U_full, "V": V_full, "W": W_full}
    
    return best_result
# EVOLVE-BLOCK-END
