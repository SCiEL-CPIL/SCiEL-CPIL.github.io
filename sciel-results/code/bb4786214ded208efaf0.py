import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    best_result = None
    best_rank = None

    # Try ranks from lower bound upwards
    # Lower bound is max(n*m, m*p, p*n) = 21
    for rank in range(21, 45):
        if time.time() > deadline - 10:
            break
            
        found = False
        # More restarts for lower ranks
        max_restarts = 100 if rank <= 25 else 50
        if time.time() > deadline - 20:
            max_restarts = 10
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            result = _try_rank_sa(rank, n, m, p, restart, deadline)
            if result is not None:
                found = True
                if best_result is None or rank < best_rank:
                    best_rank = rank
                    best_result = result
                break
        
        if found:
            # Found a valid decomposition at this rank
            # We could try to find a lower rank, but we are iterating up.
            # If we found rank 21, we are done.
            if best_rank == 21:
                break
        else:
            # If not found at this rank, continue to higher rank?
            # Usually, if you can't find rank 21, you might find 22, etc.
            # But if the true minimum is 21, and we fail, we should try harder or accept higher.
            # The loop continues to higher ranks.
            pass

    # If nothing found, return schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result

def _try_rank_sa(rank, n, m, p, restart_idx, deadline):
    """Try to find a rank-rank decomposition using Simulated Annealing on integer lattice."""
    np.random.seed(42 + restart_idx * 1000)
    
    # Initialize U, V, W with random integers from {-1, 0, 1}
    U = np.random.randint(-1, 2, size=(n * m, rank))
    V = np.random.randint(-1, 2, size=(m * p, rank))
    W = np.random.randint(-1, 2, size=(p * n, rank))
    
    # Calculate initial T_est
    T_est = np.zeros((n, m, p), dtype=np.int64)
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        # T_est[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        # Vectorized:
        u3d = u_c[:, :, np.newaxis]
        v3d = v_c[np.newaxis, :, :]
        w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
        T_est += u3d * v3d * w3d
        
    target = np.ones((n, m, p), dtype=np.int64)
    current_error = np.sum((T_est - target) ** 2)
    
    if current_error == 0:
        return _format_result(U, V, W)
        
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    steps = 0
    max_steps = 500000 # Arbitrary high, will be limited by time
    
    while time.time() < deadline - 2:
        steps += 1
        if T < min_T:
            # If stuck, maybe restart with new random init? 
            # For now, just break and return None if not found
            break
            
        # Pick a random variable to change
        var_type = np.random.randint(0, 3) # 0: U, 1: V, 2: W
        c = np.random.randint(0, rank)
        delta = 1 if np.random.random() < 0.5 else -1
        
        old_T_est_copy = None # We will update in place and revert if needed
        
        if var_type == 0:
            # Change U
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            idx_u = i * m + j
            
            # Calculate delta error
            # T_est[i,j,k] changes by delta * V[j*p+k, c] * W[k*n+i, c]
            # Let's compute the change in error
            # E = sum (T - 1)^2
            # dE = sum_{k} [ (T_ik + d_ik - 1)^2 - (T_ik - 1)^2 ]
            # d_ik = delta * V[j*p+k, c] * W[k*n+i, c]
            
            old_vals = T_est[i, j, :]
            new_vals = old_vals + delta * (V[:, c] * W[:, c]) # Wait, indices
            # V index: j*p+k. So V[j*p : j*p+p, c]
            # W index: k*n+i. So W[k*n+i, c]
            
            v_slice = V[j*p : j*p+p, c]
            w_vals = np.array([W[k*n + i, c] for k in range(p)])
            
            change_vec = delta * v_slice * w_vals
            
            old_err_slice = np.sum((old_vals - 1) ** 2)
            new_err_slice = np.sum((old_vals + change_vec - 1) ** 2)
            
            delta_error = new_err_slice - old_err_slice
            
            # Acceptance
            accept = False
            if delta_error < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta_error / T)
                    if np.random.random() < prob:
                        accept = True
                        
            if accept:
                # Apply change
                U[idx_u, c] += delta
                T_est[i, j, :] += change_vec
                current_error += delta_error
            else:
                pass # Revert (do nothing)
                
        elif var_type == 1:
            # Change V
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            idx_v = j * p + k
            
            old_vals = T_est[:, j, k] # Shape (n,)
            
            # Change in T_est[i, j, k] is delta * U[i*m+j, c] * W[k*n+i, c]
            u_vals = np.array([U[i*m + j, c] for i in range(n)])
            w_vals = np.array([W[k*n + i, c] for i in range(n)])
            
            change_vec = delta * u_vals * w_vals
            
            old_err_slice = np.sum((old_vals - 1) ** 2)
            new_err_slice = np.sum((old_vals + change_vec - 1) ** 2)
            
            delta_error = new_err_slice - old_err_slice
            
            accept = False
            if delta_error < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta_error / T)
                    if np.random.random() < prob:
                        accept = True
                        
            if accept:
                V[idx_v, c] += delta
                T_est[:, j, k] += change_vec
                current_error += delta_error
                
        else:
            # Change W
            k = np.random.randint(0, p)
            i = np.random.randint(0, n)
            idx_w = k * n + i
            
            old_vals = T_est[i, :, k] # Shape (m,)
            
            # Change in T_est[i, j, k] is delta * U[i*m+j, c] * V[j*p+k, c]
            u_vals = np.array([U[i*m + j, c] for j in range(m)])
            v_vals = np.array([V[j*p + k, c] for j in range(m)])
            
            change_vec = delta * u_vals * v_vals
            
            old_err_slice = np.sum((old_vals - 1) ** 2)
            new_err_slice = np.sum((old_vals + change_vec - 1) ** 2)
            
            delta_error = new_err_slice - old_err_slice
            
            accept = False
            if delta_error < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta_error / T)
                    if np.random.random() < prob:
                        accept = True
                        
            if accept:
                W[idx_w, c] += delta
                T_est[i, :, k] += change_vec
                current_error += delta_error
                
        # Cooling
        T *= cooling_rate
        
        if current_error == 0:
            return _format_result(U, V, W)
            
    return None

def _format_result(U, V, W):
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
