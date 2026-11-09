"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition (rank = n*m*p = 54)
    base_rank = n * m * p
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    best_rank = base_rank
    best_result = (best_U, best_V, best_W)
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact using vectorized ops."""
        rank = U.shape[1]
        # Compute all (i,j,k) sums efficiently
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        
        # For each rank component r, compute the product
        total = np.zeros(n * m * p, dtype=float)
        for r in range(rank):
            total += U[u_rows, r] * V[v_rows, r] * W[w_rows, r]
        
        if np.any(np.abs(total - 1.0) > 1e-9):
            return False
        return True
    
    def try_random_decomposition(target_rank, num_trials, value_range=(-2, 3)):
        """Try to find a valid decomposition with given rank."""
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            U = np.random.randint(value_range[0], value_range[1], size=(n * m, target_rank)).astype(float)
            V = np.random.randint(value_range[0], value_range[1], size=(m * p, target_rank)).astype(float)
            W = np.random.randint(value_range[0], value_range[1], size=(p * n, target_rank)).astype(float)
            
            if verify_decomposition(U, V, W):
                return (U, V, W)
        
        return None
    
    # Try to find lower rank decompositions
    np.random.seed(42)
    
    # Try ranks from 27 upward
    for target_rank in range(27, base_rank):
        if time.time() > deadline - 15:
            break
        
        # Number of trials decreases as rank increases (easier to find)
        if target_rank < 35:
            max_trials = 50
        elif target_rank < 45:
            max_trials = 100
        else:
            max_trials = 200
        
        result = try_random_decomposition(target_rank, max_trials)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    # If no better found, try with half-integers (multiply by 2)
    if best_rank == base_rank and time.time() < deadline - 20:
        for target_rank in range(27, base_rank):
            if time.time() > deadline - 15:
                break
            
            if target_rank < 35:
                max_trials = 50
            elif target_rank < 45:
                max_trials = 100
            else:
                max_trials = 200
            
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                
                # Use half-integers: values in {-1, -0.5, 0, 0.5, 1, 1.5, 2}
                U = np.random.randint(-4, 5, size=(n * m, target_rank)).astype(float) / 2.0
                V = np.random.randint(-4, 5, size=(m * p, target_rank)).astype(float) / 2.0
                W = np.random.randint(-4, 5, size=(p * n, target_rank)).astype(float) / 2.0
                
                if verify_decomposition(U, V, W):
                    best_rank = target_rank
                    best_result = (U, V, W)
                    break
            
            if best_rank < base_rank:
                break
    
    # Convert best result to list format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
