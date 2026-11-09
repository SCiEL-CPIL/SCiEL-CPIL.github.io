import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    T_target = np.ones((n, m, p))
    
    # Function to compute error efficiently using einsum
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by k*n+i, so W_r[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        err = np.sum((T - T_target)**2)
        return err

    # Values to search over
    values = np.array([-1, -0.5, 0, 0.5, 1])
    
    rng = np.random.default_rng(42)
    
    best_solution = None
    best_rank = rank_school
    
    # Try ranks from lower bounds upwards
    # Known lower bound for 2x3x5 is likely around 20-23. 
    # Let's try 20 to 29.
    for rank in range(20, 30):
        if time.time() > deadline - 5.0:
            break
            
        # Multiple restarts
        for attempt in range(100):
            if time.time() > deadline - 1.0:
                break
                
            # Random initialization
            U = rng.choice(values, size=(n*m, rank))
            V = rng.choice(values, size=(m*p, rank))
            W = rng.choice(values, size=(p*n, rank))
            
            err = compute_error(U, V, W)
            
            if err < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            
            # Simulated Annealing / Local Search
            current_temp = 1.0
            min_temp = 1e-6
            cooling = 0.95
            
            for step in range(2000):
                if time.time() > deadline - 0.5:
                    break
                    
                if current_temp < min_temp:
                    break
                    
                # Pick a random matrix and a random entry
                mat_idx = rng.integers(0, 3)
                if mat_idx == 0:
                    mat = U
                    size = n*m
                elif mat_idx == 1:
                    mat = V
                    size = m*p
                else:
                    mat = W
                    size = p*n
                    
                idx = rng.integers(0, size)
                r_idx = rng.integers(0, rank)
                
                old_val = mat[idx, r_idx]
                new_val = rng.choice(values)
                
                if new_val == old_val:
                    continue
                    
                mat[idx, r_idx] = new_val
                new_err = compute_error(U, V, W)
                
                delta = new_err - err
                if delta < 0 or rng.random() < np.exp(-delta / current_temp):
                    err = new_err
                    if err < 1e-9:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                else:
                    mat[idx, r_idx] = old_val
                    
                current_temp *= cooling
                
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
