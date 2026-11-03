"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def is_clean_half_int(val, tol=1e-9):
        val2 = val * 2.0
        nearest_int = round(val2)
        if abs(val2 - nearest_int) < tol:
            return True
        return False

    def check_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks from 24 up to 35
    for target_rank in range(24, 36):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        # Increase attempts for lower ranks
        attempts = 200 if target_rank < 28 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline - 10.0:
                break
            
            # Generate random sparse U and V
            # Entries in {-1, 0, 1}
            # Vary density
            p_nonzero_U = random.choice([0.3, 0.4, 0.5, 0.6])
            p_nonzero_V = random.choice([0.3, 0.4, 0.5, 0.6])
            
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            
            for c in range(target_rank):
                for i in range(n * m):
                    if random.random() < p_nonzero_U:
                        U[i][c] = random.choice([-1, 1])
                for j in range(m * p):
                    if random.random() < p_nonzero_V:
                        V[j][c] = random.choice([-1, 1])
            
            # Compute M[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # M shape: (n, p, target_rank)
            M = np.zeros((n, p, target_rank), dtype=int)
            
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j][c] * V[j * p + k][c]
                        M[i, k, c] = s
            
            # Check if we can solve for W
            # For each (i, k), we need sum_c M[i, k, c] * W[k, i, c] = 1
            # We look for a sparse solution: one non-zero W entry per (i,k)
            # W[k, i, c] must be half-integer.
            # So M[i, k, c] must be such that 1/M[i, k, c] is half-integer.
            # M is integer. So M must be +/- 1 or +/- 2.
            
            W_candidate = np.zeros((p * n, target_rank), dtype=float)
            valid_W = True
            
            for i in range(n):
                for k in range(p):
                    found_c = False
                    for c in range(target_rank):
                        m_val = M[i, k, c]
                        if m_val == 1 or m_val == -1 or m_val == 2 or m_val == -2:
                            w_val = 1.0 / m_val
                            W_candidate[k * n + i][c] = w_val
                            found_c = True
                            break
                    if not found_c:
                        valid_W = False
                        break
                if not valid_W:
                    break
            
            if valid_W:
                # Convert to lists for check and return
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_candidate.tolist()
                
                if check_decomposition(U_list, V_list, W_list, target_rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
