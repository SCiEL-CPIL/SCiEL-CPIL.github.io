import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T is sparse.
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    num_target = len(target_indices)
    
    # Precompute target tensor as a dense array for fast access?
    # Or just a set for checking?
    # For error calculation, we need to compare T_recon with T_target.
    # T_target is 1 at target_indices, 0 elsewhere.
    
    # Efficient error update structure:
    # We maintain T_recon (dim_u, dim_v, dim_w)
    # And we need to compute the change in error when a factor element changes.
    
    def compute_error(U, V, W, T_recon):
        # T_recon is already updated
        # Error = sum (T_recon - T_target)^2
        # T_target is 0 everywhere except target_indices where it is 1.
        # So Error = sum_{not target} T_recon^2 + sum_{target} (T_recon - 1)^2
        # sum_{not target} T_recon^2 = sum_all T_recon^2 - sum_target T_recon^2
        # sum_target (T_recon - 1)^2 = sum_target (T_recon^2 - 2*T_recon + 1)
        
        # We can compute sum_all T_recon^2 quickly if we maintain it?
        # Or just compute it. 576 elements.
        
        # Let's just compute the full error. It's small.
        # But we can be smarter.
        
        # For now, simple full computation for correctness in the helper.
        # In the main loop, we will use incremental updates.
        
        # T_target_vals at target_indices are 1.
        # We need sum over all (u,v,w) of (T_recon[u,v,w] - T_target[u,v,w])^2
        
        # Create a view of T_target?
        # It's better to just loop over target indices for the "on" part and use sum of squares for "off" part.
        
        # sum_off = np.sum(T_recon**2) - sum_target T_recon^2
        # sum_on = sum_target (T_recon - 1)^2
        
        # We will maintain T_recon.
        
        # Let's write a helper that takes T_recon and returns error.
        # To make it fast, we precompute a mask.
        pass

    # Let's restructure for speed.
    # We will maintain T_recon.
    # We will maintain sum_sq_all = np.sum(T_recon**2)
    # We will maintain sum_sq_target = sum of T_recon[u,v,w]**2 for u,v,w in target_indices
    # We will maintain sum_target = sum of T_recon[u,v,w] for u,v,w in target_indices
    
    # Error = (sum_sq_all - sum_sq_target) + (sum_sq_target - 2*sum_target + num_target)
    # Error = sum_sq_all - 2*sum_target + num_target
    
    # This is very efficient!
    # We just need to update sum_sq_all, sum_sq_target, sum_target when T_recon changes.
    
    def get_initial_error_stats(T_recon, target_indices):
        sum_sq_all = np.sum(T_recon**2)
        sum_sq_target = 0
        sum_target = 0
        for u, v, w in target_indices:
            val = T_recon[u, v, w]
            sum_sq_target += val * val
            sum_target += val
        # Error = sum_sq_all - 2*sum_target + num_target
        # Note: sum_sq_target cancels out!
        # Let's verify:
        # E = sum_off (T)^2 + sum_on (T-1)^2
        # E = (sum_all T^2 - sum_on T^2) + sum_on (T^2 - 2T + 1)
        # E = sum_all T^2 - sum_on T^2 + sum_on T^2 - 2 sum_on T + sum_on 1
        # E = sum_all T^2 - 2 sum_on T + num_target
        # Yes, sum_sq_target is not needed for the total error value, 
        # but we might need it if we want to check if it's exactly 0?
        # If E=0, then sum_all T^2 = 2 sum_on T - num_target.
        # Since T are integers (or half-integers), and target is 1,
        # if T is integer, T^2 >= 0.
        
        return sum_sq_all, sum_target

    def construct_solution(r, max_time_per_rank=5.0):
        """
        Attempts to find a solution of rank r using Simulated Annealing.
        Returns (U, V, W) if found, else None.
        """
        start_time = time.time()
        
        # Number of restarts
        num_restarts = 20
        
        for restart in range(num_restarts):
            if time.time() - start_time > max_time_per_rank:
                break
                
            # Initialize U, V, W
            # Random integers in {-1, 0, 1}
            # Sparsity helps avoid trivial solutions and speeds up convergence?
            # Let's try density 0.5
            
            U = np.random.choice([-1, 0, 1], size=(dim_u, r))
            V = np.random.choice([-1, 0, 1], size=(dim_v, r))
            W = np.random.choice([-1, 0, 1], size=(dim_w, r))
            
            # Compute initial T_recon
            T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
            for k in range(r):
                T_recon += U[:, k][:, None, None] * V[:, k][None, :, None] * W[:, k][None, None, :]
                
            sum_sq_all, sum_target = get_initial_error_stats(T_recon, target_indices)
            current_error = sum_sq_all - 2 * sum_target + num_target
            
            if current_error == 0:
                return U, V, W
                
            # Simulated Annealing Parameters
            T_temp = 10.0
            cooling_rate = 0.995
            min_temp = 0.01
            steps = 5000
            
            for step in range(steps):
                if time.time() - start_time > max_time_per_rank:
                    break
                    
                # Pick a random variable to change
                # 0: U, 1: V, 2: W
                mat_idx = np.random.randint(3)
                
                # Pick row and column
                if mat_idx == 0:
                    row = np.random.randint(dim_u)
                    col = np.random.randint(r)
                    # Old value
                    old_val = U[row, col]
                    # New value: change by +/- 1 or flip?
                    # Let's try adding a random delta from {-1, 1}
                    # Or just flip sign?
                    # Changing by 1 allows moving between integers.
                    delta = np.random.choice([-1, 1])
                    new_val = old_val + delta
                    # Optional: keep values small? 
                    # If values grow too large, error might explode.
                    # Let's constrain to [-2, 2] or so?
                    # Or just let it be.
                    
                    # Calculate Delta Error
                    # T_recon changes by delta * V[:, col] * W[:, col]
                    # Let vec_v = V[:, col], vec_w = W[:, col]
                    # Change in T_recon at (u, v, w) is delta * V[v, col] * W[w, col]
                    # Only u=row is affected in U, but V and W are full vectors.
                    # So T_recon[row, :, :] changes by delta * vec_v[:, None] * vec_w[None, :]
                    
                    # We need to update sum_sq_all and sum_target.
                    # Let's compute the change in T_recon for the affected slice.
                    # Affected slice: T_recon[row, :, :]
                    
                    # Old slice: T_old = T_recon[row, :, :]
                    # New slice: T_new = T_old + delta * outer(vec_v, vec_w)
                    
                    # Change in sum_sq_all:
                    # sum_new_sq - sum_old_sq = sum(T_new^2 - T_old^2)
                    # = sum((T_old + D)^2 - T_old^2) where D = delta * outer(vec_v, vec_w)
                    # = sum(2 * T_old * D + D^2)
                    
                    D = delta * np.outer(vec_v, vec_w)
                    T_old_slice = T_recon[row, :, :]
                    
                    delta_sum_sq_all = np.sum(2 * T_old_slice * D + D * D)
                    
                    # Change in sum_target:
                    # Only target indices in this slice contribute.
                    delta_sum_target = 0
                    for u, v, w in target_indices:
                        if u == row:
                            # T_recon[u, v, w] changes by D[v, w]
                            delta_sum_target += D[v, w]
                    
                    new_sum_sq_all = sum_sq_all + delta_sum_sq_all
                    new_sum_target = sum_target + delta_sum_target
                    new_error = new_sum_sq_all - 2 * new_sum_target + num_target
                    
                    # Acceptance criterion
                    if new_error < current_error:
                        # Accept
                        U[row, col] = new_val
                        T_recon[row, :, :] += D
                        sum_sq_all = new_sum_sq_all
                        sum_target = new_sum_target
                        current_error = new_error
                        if current_error == 0:
                            return U, V, W
                    else:
                        # Metropolis criterion
                        if np.random.rand() < np.exp((current_error - new_error) / T_temp):
                            U[row, col] = new_val
                            T_recon[row, :, :] += D
                            sum_sq_all = new_sum_sq_all
                            sum_target = new_sum_target
                            current_error = new_error
                            
                elif mat_idx == 1:
                    row = np.random.randint(dim_v)
                    col = np.random.randint(r)
                    old_val = V[row, col]
                    delta = np.random.choice([-1, 1])
                    new_val = old_val + delta
                    
                    # Affected: T_recon[:, row, :]
                    vec_u = U[:, col]
                    vec_w = W[:, col]
                    D = delta * np.outer(vec_u, vec_w) # Shape (dim_u, dim_w)
                    # T_recon[:, row, :] += D
                    # Wait, D is (dim_u, dim_w). T_recon is (dim_u, dim_v, dim_w).
                    # T_recon[:, row, :] is (dim_u, dim_w). Correct.
                    
                    T_old_slice = T_recon[:, row, :]
                    delta_sum_sq_all = np.sum(2 * T_old_slice * D + D * D)
                    
                    delta_sum_target = 0
                    for u, v, w in target_indices:
                        if v == row:
                            delta_sum_target += D[u, w]
                            
                    new_sum_sq_all = sum_sq_all + delta_sum_sq_all
                    new_sum_target = sum_target + delta_sum_target
                    new_error = new_sum_sq_all - 2 * new_sum_target + num_target
                    
                    if new_error < current_error or np.random.rand() < np.exp((current_error - new_error) / T_temp):
                        V[row, col] = new_val
                        T_recon[:, row, :] += D
                        sum_sq_all = new_sum_sq_all
                        sum_target = new_sum_target
                        current_error = new_error
                        if current_error == 0:
                            return U, V, W
                            
                else: # mat_idx == 2
                    row = np.random.randint(dim_w)
                    col = np.random.randint(r)
                    old_val = W[row, col]
                    delta = np.random.choice([-1, 1])
                    new_val = old_val + delta
                    
                    # Affected: T_recon[:, :, row]
                    vec_u = U[:, col]
                    vec_v = V[:, col]
                    D = delta * np.outer(vec_u, vec_v) # Shape (dim_u, dim_v)
                    # T_recon[:, :, row] += D
                    # T_recon[:, :, row] is (dim_u, dim_v). Correct.
                    
                    T_old_slice = T_recon[:, :, row]
                    delta_sum_sq_all = np.sum(2 * T_old_slice * D + D * D)
                    
                    delta_sum_target = 0
                    for u, v, w in target_indices:
                        if w == row:
                            delta_sum_target += D[u, v]
                            
                    new_sum_sq_all = sum_sq_all + delta_sum_sq_all
                    new_sum_target = sum_target + delta_sum_target
                    new_error = new_sum_sq_all - 2 * new_sum_target + num_target
                    
                    if new_error < current_error or np.random.rand() < np.exp((current_error - new_error) / T_temp):
                        W[row, col] = new_val
                        T_recon[:, :, row] += D
                        sum_sq_all = new_sum_sq_all
                        sum_target = new_sum_target
                        current_error = new_error
                        if current_error == 0:
                            return U, V, W
                            
                # Cool down
                T_temp *= cooling_rate
                if T_temp < min_temp:
                    break
                    
        return None

    # Try ranks from low to high
    # Lower bound estimation:
    # For 2x2x2, rank 7.
    # For 2x3x4, let's start at 16.
    # If not found, try 17, 18, etc.
    
    for r in range(16, 25):
        sol = construct_solution(r, max_time_per_rank=5.0)
        if sol is not None:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, column] = 1
                V[v, column] = 1
                W[w, column] = 1
                column += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
