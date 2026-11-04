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
    
    # Precompute T as float for faster comparison
    T_float = T.astype(float)
    
    def fast_check(U_np, V_np, W_np, rank):
        """Fast verification using matrix operations."""
        # T_est[x,y,z] = sum_r U[x,r] V[y,r] W[z,r]
        # For each r, compute outer product and sum
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('x,y,z->xyz', U_np[:, r], V_np[:, r], W_np[:, r])
        return np.allclose(T_est, T_float, atol=1e-9)
    
    def fast_check_partial(U_np, V_np, W_np, rank, target_rank):
        """Check if current state is close to valid (for guiding search)."""
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('x,y,z->xyz', U_np[:, r], V_np[:, r], W_np[:, r])
        # Return L2 error
        return np.linalg.norm(T_est - T_float)
    
    # Try rank 27, 26, 25...
    for target_rank in range(rank_known - 1, 19, -1):
        if time.time() > deadline:
            break
        
        found = False
        num_restarts = 30 if target_rank >= 26 else 20
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if target_rank == rank_known - 1:
                # Try removing one column and compensating with linear combinations
                for col_to_remove in range(rank_known):
                    if time.time() > deadline:
                        break
                    
                    # Create a copy without this column
                    mask = [i for i in range(rank_known) if i != col_to_remove]
                    U_sub = U_known_np[:, mask].copy()
                    V_sub = V_known_np[:, mask].copy()
                    W_sub = W_known_np[:, mask].copy()
                    
                    # Check if this alone works (unlikely)
                    if fast_check(U_sub, V_sub, W_sub, target_rank):
                        best_rank = target_rank
                        best_U = U_sub.tolist()
                        best_V = V_sub.tolist()
                        best_W = W_sub.tolist()
                        found = True
                        break
                    
                    # Try structured perturbations: modify entries in a pattern
                    # Focus on columns that interact with the removed column
                    for pert_iter in range(50):
                        if time.time() > deadline:
                            break
                        
                        U_pert = U_sub.copy()
                        V_pert = V_sub.copy()
                        W_pert = W_sub.copy()
                        
                        # Apply structured perturbations based on the removed column's pattern
                        # The removed column had certain non-zero entries; try to redistribute
                        removed_U = U_known_np[:, col_to_remove]
                        removed_V = V_known_np[:, col_to_remove]
                        removed_W = W_known_np[:, col_to_remove]
                        
                        # Find which rows had non-zero entries in removed column
                        U_nonzero = np.nonzero(removed_U)[0]
                        V_nonzero = np.nonzero(removed_V)[0]
                        W_nonzero = np.nonzero(removed_W)[0]
                        
                        # Try to compensate by modifying existing columns
                        for comp_col in random.sample(range(target_rank), min(3, target_rank)):
                            if len(U_nonzero) > 0:
                                U_pert[random.choice(U_nonzero), comp_col] += random.choice([-1, 1])
                            if len(V_nonzero) > 0:
                                V_pert[random.choice(V_nonzero), comp_col] += random.choice([-1, 1])
                            if len(W_nonzero) > 0:
                                W_pert[random.choice(W_nonzero), comp_col] += random.choice([-1, 1])
                        
                        if fast_check(U_pert, V_pert, W_pert, target_rank):
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
                # For lower ranks, use structured random search with block awareness
                U_try = np.zeros((8, target_rank))
                V_try = np.zeros((16, target_rank))
                W_try = np.zeros((8, target_rank))
                
                # Initialize with block-structured patterns
                for r in range(target_rank):
                    # Each term should contribute to some pattern
                    # Use the block structure: rows 0-3 are first i-block, 4-7 second
                    block = random.choice([0, 1])
                    U_rows = [block*4 + random.randint(0,3) for _ in range(2)]
                    V_rows = [random.randint(0,15) for _ in range(2)]
                    W_rows = [random.randint(0,7) for _ in range(2)]
                    
                    U_try[U_rows[0], r] = random.choice([-1, 1])
                    U_try[U_rows[1], r] = random.choice([-1, 1])
                    V_try[V_rows[0], r] = random.choice([-1, 1])
                    V_try[V_rows[1], r] = random.choice([-1, 1])
                    W_try[W_rows[0], r] = random.choice([-1, 1])
                    W_try[W_rows[1], r] = random.choice([-1, 1])
                
                # Local search with gradient-guided steps
                best_err = fast_check_partial(U_try, V_try, W_try, target_rank, target_rank)
                for step in range(200):
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
                    if fast_check(U_try, V_try, W_try, target_rank):
                        best_rank = target_rank
                        best_U = U_try.tolist()
                        best_V = V_try.tolist()
                        best_W = W_try.tolist()
                        found = True
                        break
                    
                    # Track best error for potential restart
                    curr_err = fast_check_partial(U_try, V_try, W_try, target_rank, target_rank)
                    if curr_err < best_err:
                        best_err = curr_err
        
        if found:
            break  # Found a solution at this rank, no need to try lower ranks

    return {"U": best_U, "V": best_V, "W": best_W}
