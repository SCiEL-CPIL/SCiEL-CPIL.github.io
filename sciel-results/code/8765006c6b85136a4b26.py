"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Try ranks from 12 up to schoolbook_rank - 1
    for rank in range(12, min(schoolbook_rank, 40)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if rank <= 15 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U = np.random.uniform(-1, 1, (n * m, rank))
            V = np.random.uniform(-1, 1, (m * p, rank))
            W = np.random.uniform(-1, 1, (p * n, rank))
            
            # Gradient-free local search: iteratively perturb and accept if residual decreases
            def compute_residual(U, V, W):
                # For each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            res += (val - 1.0) ** 2
                return res
            
            current_res = compute_residual(U, V, W)
            
            # Iterative refinement
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                if current_res < 1e-10:
                    break
                
                # Pick a random factor and a random entry to perturb
                factor_choice = random.randint(0, 2)
                if factor_choice == 0:
                    r_idx = random.randint(0, n * m - 1)
                    c_idx = random.randint(0, rank - 1)
                    old_val = U[r_idx, c_idx]
                    U[r_idx, c_idx] = old_val + random.gauss(0, 0.1)
                elif factor_choice == 1:
                    r_idx = random.randint(0, m * p - 1)
                    c_idx = random.randint(0, rank - 1)
                    old_val = V[r_idx, c_idx]
                    V[r_idx, c_idx] = old_val + random.gauss(0, 0.1)
                else:
                    r_idx = random.randint(0, p * n - 1)
                    c_idx = random.randint(0, rank - 1)
                    old_val = W[r_idx, c_idx]
                    W[r_idx, c_idx] = old_val + random.gauss(0, 0.1)
                
                new_res = compute_residual(U, V, W)
                
                if new_res < current_res:
                    current_res = new_res
                else:
                    # Revert
                    if factor_choice == 0:
                        U[r_idx, c_idx] = old_val
                    elif factor_choice == 1:
                        V[r_idx, c_idx] = old_val
                    else:
                        W[r_idx, c_idx] = old_val
            
            # Check if we found a valid solution
            if current_res < 1e-6:
                # Snap to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify exactness
                final_res = compute_residual(U, V, W)
                if final_res < 1e-6:
                    # Convert to lists
                    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
                    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
                    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
                    
                    if rank < best_rank:
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        break  # Found minimal for this rank, try lower
        
        # If we found a valid decomposition at this rank, we can stop if rank is low
        if best_rank <= 12:
            break
    
    return best
# EVOLVE-BLOCK-END
