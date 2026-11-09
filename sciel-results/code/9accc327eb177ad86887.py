import time
import numpy as np
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition
    rank_school = n * m * p  # 72
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school
    
    # Strategy: Split 3x3x8 into two 3x3x4 problems.
    # Find a low rank decomposition for 3x3x4, then duplicate it.
    # Target ranks for 3x3x4: 24, 25, 26, 27, 28, 29, 30
    # Corresponding ranks for 3x3x8: 48, 50, 52, 54, 56, 58, 60
    
    # We will search for a decomposition of 3x3x4 with rank R.
    # If found, we construct 3x3x8 with rank 2R.
    
    def solve_334(target_rank, time_limit):
        """
        Find a rank-target_rank decomposition for 3x3x4 matrix multiplication tensor.
        Returns (U, V, W) or None.
        """
        n2, m2, p2 = 3, 3, 4
        T_target = np.ones((n2, m2, p2))
        
        U = np.zeros((n2 * m2, target_rank))
        V = np.zeros((m2 * p2, target_rank))
        W = np.zeros((p2 * n2, target_rank))
        
        # Initialize with random integers
        random.seed(42)
        for r in range(target_rank):
            U[:, r] = np.random.choice([-1, 0, 1], size=n2 * m2)
            V[:, r] = np.random.choice([-1, 0, 1], size=m2 * p2)
            W[:, r] = np.random.choice([-1, 0, 1], size=p2 * n2)
            
            # Ensure at least one non-zero in each vector to avoid trivial zeros
            if np.all(U[:, r] == 0):
                U[np.random.randint(n2 * m2), r] = 1
            if np.all(V[:, r] == 0):
                V[np.random.randint(m2 * p2), r] = 1
            if np.all(W[:, r] == 0):
                W[np.random.randint(p2 * n2), r] = 1
        
        def calc_error(U, V, W):
            err = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0
                        for r in range(target_rank):
                            val += U[i * m2 + j, r] * V[j * p2 + k, r] * W[k * n2 + i, r]
                        err += (1.0 - val) ** 2
            return err
        
        current_error = calc_error(U, V, W)
        if current_error < 1e-9:
            return U, V, W
            
        # Simulated Annealing / Local Search
        T = 1.0
        cooling = 0.99
        steps = 0
        max_steps = 50000
        start_search = time.time()
        
        while steps < max_steps and time.time() - start_search < time_limit:
            steps += 1
            # Pick a random variable to change
            var_type = random.choice(['U', 'V', 'W'])
            idx1 = random.randint(0, target_rank - 1)
            if var_type == 'U':
                idx2 = random.randint(0, n2 * m2 - 1)
                old_val = U[idx2, idx1]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                U[idx2, idx1] = new_val
                new_error = calc_error(U, V, W)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_error = new_error
                    if current_error < 1e-9:
                        return U, V, W
                else:
                    U[idx2, idx1] = old_val
            elif var_type == 'V':
                idx2 = random.randint(0, m2 * p2 - 1)
                old_val = V[idx2, idx1]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                V[idx2, idx1] = new_val
                new_error = calc_error(U, V, W)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_error = new_error
                    if current_error < 1e-9:
                        return U, V, W
                else:
                    V[idx2, idx1] = old_val
            else: # W
                idx2 = random.randint(0, p2 * n2 - 1)
                old_val = W[idx2, idx1]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                W[idx2, idx1] = new_val
                new_error = calc_error(U, V, W)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_error = new_error
                    if current_error < 1e-9:
                        return U, V, W
                else:
                    W[idx2, idx1] = old_val
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return None

    # Try ranks for 3x3x4
    # Start with 24 (best case for 3x3x8 is 48)
    # If 24 fails, try 25 (50), 26 (52), etc.
    for r334 in range(24, 31):
        if time.time() > deadline - 20:
            break
            
        time_per_try = min(10, max(2, (deadline - 20 - time.time()) / (31 - r334)))
        sol = solve_334(r334, time_per_try)
        
        if sol is not None:
            U334, V334, W334 = sol
            # Construct 3x3x8 solution
            rank_338 = 2 * r334
            U338 = np.zeros((n * m, rank_338))
            V338 = np.zeros((m * p, rank_338))
            W338 = np.zeros((p * n, rank_338))
            
            # First half (k=0..3)
            U338[:, :r334] = U334
            V338[:, :r334] = V334
            # W338 for k=0..3
            for k in range(4):
                for i in range(3):
                    W338[k * 3 + i, :r334] = W334[k * 3 + i]
                    
            # Second half (k=4..7)
            # Shift indices for V and W
            U338[:, r334:] = U334
            for k in range(4):
                for j in range(3):
                    # V[j, k+4] corresponds to V334[j, k]
                    V338[j * 8 + (k + 4), r334:] = V334[j * 4 + k]
            for k in range(4):
                for i in range(3):
                    # W[k+4, i] corresponds to W334[k, i]
                    W338[(k + 4) * 3 + i, r334:] = W334[k * 3 + i]
                    
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank_338):
                            val += U338[i * m + j, r] * V338[j * p + k, r] * W338[k * n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                if rank_338 < best_rank:
                    best_U = U338
                    best_V = V338
                    best_W = W338
                    best_rank = rank_338
                break # Found a valid one, stop searching for lower ranks in this loop? 
                      # Actually, we want the lowest rank. We started from 24.
                      # If we found 24, we break. If we found 25, we break.
                      # So we break as soon as we find a valid one.
    
    # If no better rank found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
