import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def calc_S(U, V, W):
        # U: (rank, n, m), V: (rank, m, p), W: (rank, n, p)
        # S[i,j,k] = sum_s U[s,i,j] * V[s,j,k] * W[s,i,k]
        return np.einsum('sij,sjk,sik->ijk', U, V, W)
    
    def get_rank_30_solution():
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # We will try ranks from 15 up to 30
    # For each rank, we use a local search to minimize the squared error
    # But first, let's try a more structured approach if possible.
    # The task asks for rank-minimal.
    # Known lower bound for R(2,3,5) is at least 15 (n*m = 6, m*p = 15, n*p = 10, max is 15? No, lower bound is max(n*m, m*p, n*p) is not correct for bilinear complexity usually, but for standard matrix multiplication R(m,n,p) >= max(mn, np, pm). Here max(6, 10, 15) = 15.
    # So rank 15 is the theoretical lower bound.
    
    best_result = None
    best_rank = 31
    
    # Try to find a solution for rank 15 first, then 16, etc.
    # Given the difficulty, we might need a very good heuristic.
    # Let's use a simulated annealing-like local search with better restarts.
    
    def local_search(rank, max_iters, deadline, seed):
        np.random.seed(seed)
        
        # Initialize
        # Try a few different initialization strategies
        strategy = np.random.randint(3)
        
        if strategy == 0:
            # Random half-integers
            vals = np.arange(-2, 2.5, 0.5)
            U = np.random.choice(vals, size=(rank, n, m))
            V = np.random.choice(vals, size=(rank, m, p))
            W = np.random.choice(vals, size=(rank, n, p))
        elif strategy == 1:
            # Sparse random
            U = np.zeros((rank, n, m))
            V = np.zeros((rank, m, p))
            W = np.zeros((rank, n, p))
            for s in range(rank):
                # Pick a few non-zeros
                for _ in range(2):
                    i, j = np.random.randint(n), np.random.randint(m)
                    U[s, i, j] = np.random.choice([-1, 1, 0.5, -0.5])
                for _ in range(2):
                    j, k = np.random.randint(m), np.random.randint(p)
                    V[s, j, k] = np.random.choice([-1, 1, 0.5, -0.5])
                for _ in range(2):
                    i, k = np.random.randint(n), np.random.randint(p)
                    W[s, i, k] = np.random.choice([-1, 1, 0.5, -0.5])
        else:
            # Structured: try to mimic the rank 30 solution but with overlaps
            # This is hard to do simply. Let's just use random but with larger values sometimes.
            vals = np.arange(-5, 5.5, 0.5)
            U = np.random.choice(vals, size=(rank, n, m))
            V = np.random.choice(vals, size=(rank, m, p))
            W = np.random.choice(vals, size=(rank, n, p))
            
        S = calc_S(U, V, W)
        error = np.sum((S - 1.0)**2)
        
        # Local search loop
        iter_count = 0
        temp = 1.0
        cooling = 0.999
        
        while error > 1e-9 and iter_count < max_iters:
            if time.time() > deadline:
                break
                
            iter_count += 1
            
            # Choose tensor to update
            tensor_id = np.random.choice(3)
            
            if tensor_id == 0:
                s, i, j = np.random.randint(rank), np.random.randint(n), np.random.randint(m)
                old_val = U[s, i, j]
                # Step size
                step = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + step
                if new_val < -10 or new_val > 10:
                    continue
                
                delta = new_val - old_val
                current_S_row = S[i, j, :]
                new_S_row = current_S_row + delta * (V[s, j, :] * W[s, i, :])
                
                old_err_part = np.sum((current_S_row - 1.0)**2)
                new_err_part = np.sum((new_S_row - 1.0)**2)
                delta_err = new_err_part - old_err_part
                
                if delta_err < 0 or np.random.rand() < np.exp(-delta_err / (temp + 1e-9)):
                    U[s, i, j] = new_val
                    S[i, j, :] = new_S_row
                    error += delta_err
                    
            elif tensor_id == 1:
                s, j, k = np.random.randint(rank), np.random.randint(m), np.random.randint(p)
                old_val = V[s, j, k]
                step = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + step
                if new_val < -10 or new_val > 10:
                    continue
                
                delta = new_val - old_val
                current_S_col = S[:, j, k]
                new_S_col = current_S_col + delta * (U[s, :, j] * W[s, :, k])
                
                old_err_part = np.sum((current_S_col - 1.0)**2)
                new_err_part = np.sum((new_S_col - 1.0)**2)
                delta_err = new_err_part - old_err_part
                
                if delta_err < 0 or np.random.rand() < np.exp(-delta_err / (temp + 1e-9)):
                    V[s, j, k] = new_val
                    S[:, j, k] = new_S_col
                    error += delta_err
                    
            else:
                s, i, k = np.random.randint(rank), np.random.randint(n), np.random.randint(p)
                old_val = W[s, i, k]
                step = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + step
                if new_val < -10 or new_val > 10:
                    continue
                
                delta = new_val - old_val
                current_S_vec = S[i, :, k]
                new_S_vec = current_S_vec + delta * (U[s, i, :] * V[s, :, k])
                
                old_err_part = np.sum((current_S_vec - 1.0)**2)
                new_err_part = np.sum((new_S_vec - 1.0)**2)
                delta_err = new_err_part - old_err_part
                
                if delta_err < 0 or np.random.rand() < np.exp(-delta_err / (temp + 1e-9)):
                    W[s, i, k] = new_val
                    S[i, :, k] = new_S_vec
                    error += delta_err
            
            temp *= cooling
        
        if error < 1e-9:
            # Found a solution!
            U_list = [[U[s, i, j] for s in range(rank)] for i in range(n) for j in range(m)]
            V_list = [[V[s, j, k] for s in range(rank)] for j in range(m) for k in range(p)]
            W_list = [[W[s, i, k] for s in range(rank)] for k in range(p) for i in range(n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    # Search ranks from 15 upwards
    # We have limited time, so we can't do exhaustive search for each.
    # Let's try rank 15 with a few attempts, then 16, etc.
    
    # To save time, we will allocate time based on rank.
    # Rank 15 is the most important.
    
    ranks_to_try = list(range(15, 31))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Number of attempts depends on remaining time and rank priority
        # Lower ranks get more attempts initially.
        if rank == 15:
            num_attempts = 20
            max_iters_per_attempt = 20000
        elif rank == 16:
            num_attempts = 10
            max_iters_per_attempt = 15000
        else:
            num_attempts = 5
            max_iters_per_attempt = 10000
            
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
                
            result = local_search(rank, max_iters_per_attempt, deadline - 5, seed=rank*100 + attempt)
            if result is not None:
                return result
                
    # Fallback to rank 30
    return get_rank_30_solution()
