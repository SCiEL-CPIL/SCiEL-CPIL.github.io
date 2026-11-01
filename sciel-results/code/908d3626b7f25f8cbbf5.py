import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def calc_error(U, V, W, R):
        """
        Calculate the squared error of the tensor decomposition.
        U: shape (n*m, R)
        V: shape (m*p, R)
        W: shape (p*n, R)
        """
        error = 0.0
        # Main terms: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized dot product for this specific (i,j,k)
                    # U_idx = i*m + j
                    # V_idx = j*p + k
                    # W_idx = k*n + i
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    # sum(u*v*w)
                    s = np.dot(u_row * v_row, w_row)
                    error += (s - 1.0) ** 2
                    
                    # Cross terms: sum_c U[i,j,c] V[j,k,c] W[k,i',c] == 0 for i' != i
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        w_row2 = W[k*n + i2]
                        s2 = np.dot(u_row * v_row, w_row2)
                        error += s2 ** 2
        return error

    def random_half_int(low, high):
        # Random half-integer in [low, high]
        val = random.uniform(low, high)
        return round(val * 2) / 2

    def anneal(R, max_steps=20000, T_start=10.0, T_end=0.01):
        """
        Simulated Annealing to find a valid decomposition of rank R.
        """
        # Initialize with random small half-integers
        # Use a restricted range to help convergence
        U = np.array([[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(n*m)], dtype=float)
        V = np.array([[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(m*p)], dtype=float)
        W = np.array([[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(p*n)], dtype=float)
        
        # Flatten for easier manipulation
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Combine into a single list of arrays for mutation? 
        # Or just keep them separate.
        
        current_error = calc_error(U, V, W, R)
        best_error = current_error
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Total number of variables
        n_vars = n*m*R + m*p*R + p*n*R
        
        # Pre-compute indices for faster access? 
        # calc_error is fast enough for 20k steps.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (step / max_steps)
            
            # Pick a random variable to mutate
            var_idx = random.randint(0, n_vars - 1)
            
            # Determine which matrix and index
            if var_idx < n*m*R:
                mat = U
                flat_idx = var_idx
                # Get old value
                old_val = U_flat[flat_idx]
                # Propose new value: neighbor in half-integers
                # Random step size
                step_size = random.choice([0.5, -0.5, 1.0, -1.0, 1.5, -1.5])
                new_val = old_val + step_size
                # Keep values bounded to prevent explosion, e.g., [-5, 5]
                if new_val < -5 or new_val > 5:
                    continue
                # Apply mutation
                U.flat[flat_idx] = new_val
                new_error = calc_error(U, V, W, R)
                delta_e = new_error - current_error
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    # Revert
                    U.flat[flat_idx] = old_val
                    
            elif var_idx < n*m*R + m*p*R:
                mat = V
                flat_idx = var_idx - n*m*R
                old_val = V_flat[flat_idx]
                step_size = random.choice([0.5, -0.5, 1.0, -1.0, 1.5, -1.5])
                new_val = old_val + step_size
                if new_val < -5 or new_val > 5:
                    continue
                V.flat[flat_idx] = new_val
                new_error = calc_error(U, V, W, R)
                delta_e = new_error - current_error
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    V.flat[flat_idx] = old_val
                    
            else:
                mat = W
                flat_idx = var_idx - (n*m*R + m*p*R)
                old_val = W_flat[flat_idx]
                step_size = random.choice([0.5, -0.5, 1.0, -1.0, 1.5, -1.5])
                new_val = old_val + step_size
                if new_val < -5 or new_val > 5:
                    continue
                W.flat[flat_idx] = new_val
                new_error = calc_error(U, V, W, R)
                delta_e = new_error - current_error
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    W.flat[flat_idx] = old_val
            
            # Update flat views (optional if we use .flat directly, but good for consistency if we used views)
            # Since we modified U, V, W in place, and U_flat was a view? 
            # np.array(...).flatten() returns a copy if not contiguous? 
            # U is created as list of lists -> np.array. It is C-contiguous. 
            # U.flatten() returns a view.
            # So U_flat reflects changes.
            
            if best_error < 1e-6:
                return best_U, best_V, best_W, best_error
                
        return best_U, best_V, best_W, best_error

    def is_valid(U, V, W, R):
        """Check if error is effectively zero."""
        err = calc_error(U, V, W, R)
        return err < 1e-6

    # Try Rank 10
    best_rank = 12
    best_U, best_V, best_W = None, None, None
    
    # Fallback schoolbook
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school

    # Attempt Rank 10
    if time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W, err = anneal(10, max_steps=5000)
            if err < 1e-6:
                # Convert to list of lists of floats (half-integers)
                best_rank = 10
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
    # Attempt Rank 11 if 10 failed
    if best_rank > 11 and time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W, err = anneal(11, max_steps=5000)
            if err < 1e-6:
                best_rank = 11
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
