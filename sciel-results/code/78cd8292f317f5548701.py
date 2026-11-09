"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the target tensor T_{i,j,k} = 1 for all valid i,j,k
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k}
    # For matrix multiplication, T_{i,j,k} = 1 (each term A_{i,j}B_{j,k} appears exactly once)
    
    # Start with schoolbook
    U = np.zeros((n*m, rank_schoolbook), dtype=float)
    V = np.zeros((m*p, rank_schoolbook), dtype=float)
    W = np.zeros((p*n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to reduce rank by checking if columns are linearly dependent
    # or by randomized search for lower rank
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try ranks from schoolbook down
    for target_rank in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # For each target rank, try random initialization and refinement
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with random half-integer values
            U_rand = np.zeros((n*m, target_rank), dtype=float)
            V_rand = np.zeros((m*p, target_rank), dtype=float)
            W_rand = np.zeros((p*n, target_rank), dtype=float)
            
            # Use a structured initialization based on schoolbook
            # Map schoolbook columns to target_rank
            if target_rank == rank_schoolbook:
                U_rand = U.copy()
                V_rand = V.copy()
                W_rand = W.copy()
            else:
                # Random initialization with small values
                for r in range(target_rank):
                    # Assign random subsets
                    for idx in range(n*m):
                        if random.random() < 0.5:
                            U_rand[idx, r] = random.choice([0, 0.5, 1, 1.5, 2])
                    for idx in range(m*p):
                        if random.random() < 0.5:
                            V_rand[idx, r] = random.choice([0, 0.5, 1, 1.5, 2])
                    for idx in range(p*n):
                        if random.random() < 0.5:
                            W_rand[idx, r] = random.choice([0, 0.5, 1, 1.5, 2])
            
            # Check if this decomposition is valid
            # For each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        if abs(s - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break  # Found valid decomposition at this rank, try lower
            
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(rounded - int(rounded)) < 1e-9:
                    row_list.append(int(rounded))
                else:
                    row_list.append(rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
