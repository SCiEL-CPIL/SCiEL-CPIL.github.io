import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Current known solution
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    # Replicate the logic from the prompt to generate U_known, V_known, W_known
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def check_solution(U, V, W, rank):
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute tensor T_est
        # T_est[x, y, z] = sum_r U[x, r] V[y, r] W[z, r]
        # We can do this by summing outer products
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            # Add u[:, None, None] * v[None, :, None] * w[None, None, :]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        # Check if T_est matches T
        # Only need to check where T is 1? No, must be 0 elsewhere too.
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Start search for lower rank
    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Convert known solution to numpy arrays for efficient manipulation
    U_known_np = np.array(U_known, dtype=float)
    V_known_np = np.array(V_known, dtype=float)
    W_known_np = np.array(W_known, dtype=float)
    
    # Try rank 27, 26, 25...
    for target_rank in range(rank_known - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Strategy: Start from the known solution and try to merge/remove terms
        # Use local search with small perturbations
        
        found = False
        num_restarts = 50 if target_rank >= 26 else 30
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize by taking a subset of columns from the known solution
            # or by perturbing the known solution
            
            if target_rank == rank_known - 1:
                # Try removing one column and adjusting others
                # Start from known solution
                U_try = U_known_np.copy()
                V_try = V_known_np.copy()
                W_try = W_known_np.copy()
                
                # Try removing each column and see if we can compensate
                for col_to_remove in range(rank_known):
                    if time.time() > deadline:
                        break
                    
                    # Create a copy without this column
                    U_sub = np.delete(U_try, col_to_remove, axis=1)
                    V_sub = np.delete(V_try, col_to_remove, axis=1)
                    W_sub = np.delete(W_try, col_to_remove, axis=1)
                    
                    # Check if this alone works (unlikely)
                    T_est = np.zeros((8, 16, 8), dtype=float)
                    for r in range(target_rank):
                        T_est += np.einsum('r,s,t->rst', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                    
                    if np.allclose(T_est, T, atol=1e-9):
                        # Found a valid lower rank solution!
                        best_rank = target_rank
                        best_U = U_sub.tolist()
                        best_V = V_sub.tolist()
                        best_W = W_sub.tolist()
                        found = True
                        break
                    
                    # Try small perturbations to the remaining columns
                    for pert_iter in range(20):
                        if time.time() > deadline:
                            break
                        
                        U_pert = U_sub.copy()
                        V_pert = V_sub.copy()
                        W_pert = W_sub.copy()
                        
                        # Apply small random perturbations
                        for idx in range(3):
                            r_idx = random.randint(0, target_rank - 1)
                            U_pert[random.randint(0, 7), r_idx] += random.choice([-1, 1])
                            V_pert[random.randint(0, 15), r_idx] += random.choice([-1, 1])
                            W_pert[random.randint(0, 7), r_idx] += random.choice([-1, 1])
                        
                        # Check
                        T_est = np.zeros((8, 16, 8), dtype=float)
                        for r in range(target_rank):
                            T_est += np.einsum('r,s,t->rst', U_pert[:, r], V_pert[:, r], W_pert[:, r])
                        
                        if np.allclose(T_est, T, atol=1e-9):
                            best_rank = target_rank
                            best_U = U_pert.tolist()
                            best_V = V_pert.tolist()
                            best_W = W_pert.tolist()
                            found = True
                            break
                        
                        if found:
                            break
                    if found:
                        break
            else:
                # For lower ranks, use more aggressive random search with structure
                # Initialize with random values but with structure
                U_try = np.zeros((8, target_rank))
                V_try = np.zeros((16, target_rank))
                W_try = np.zeros((8, target_rank))
                
                # Fill with small random integers
                for r in range(target_rank):
                    # Each term should contribute to some pattern
                    U_try[random.sample(range(8), 2), r] = random.choice([-1, 1])
                    V_try[random.sample(range(16), 2), r] = random.choice([-1, 1])
                    W_try[random.sample(range(8), 2), r] = random.choice([-1, 1])
                
                # Local search
                for step in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Try modifying one entry
                    matrix_choice = random.randint(0, 2)
                    if matrix_choice == 0:
                        U_try[random.randint(0, 7), random.randint(0, target_rank - 1)] += random.choice([-1, 1])
                    elif matrix_choice == 1:
                        V_try[random.randint(0, 15), random.randint(0, target_rank - 1)] += random.choice([-1, 1])
                    else:
                        W_try[random.randint(0, 7), random.randint(0, target_rank - 1)] += random.choice([-1, 1])
                    
                    # Check
                    T_est = np.zeros((8, 16, 8), dtype=float)
                    for r in range(target_rank):
                        T_est += np.einsum('r,s,t->rst', U_try[:, r], V_try[:, r], W_try[:, r])
                    
                    if np.allclose(T_est, T, atol=1e-9):
                        best_rank = target_rank
                        best_U = U_try.tolist()
                        best_V = V_try.tolist()
                        best_W = W_try.tolist()
                        found = True
                        break
        
        if found:
            break  # Found a solution at this rank, no need to try lower ranks

    return {"U": best_U, "V": best_V, "W": best_W}
