import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape to 3D
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            val = np.dot(U_3d[i, j, :], V_3d[j_prime, k, :] * W_3d[k_prime, i_prime, :])
                            if abs(val - target) > 1e-9:
                                return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the total error of the decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return float('inf')
        
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    total_error = 0.0
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            val = np.dot(U_3d[i, j, :], V_3d[j_prime, k, :] * W_3d[k_prime, i_prime, :])
                            total_error += (val - target) ** 2
    return total_error

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _random_init(n, m, p, r, values):
    """Randomly initialize U, V, W with entries from given values."""
    U = [[random.choice(values) for _ in range(r)] for _ in range(n * m)]
    V = [[random.choice(values) for _ in range(r)] for _ in range(m * p)]
    W = [[random.choice(values) for _ in range(r)] for _ in range(p * n)]
    return U, V, W

def _local_search(U, V, W, n, m, p, r, values, max_iters=500, timeout=10):
    """Local search to reduce error by changing one entry at a time."""
    start = time.time()
    current_error = _compute_error(U, V, W, n, m, p)
    
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Pick a random variable to change
        var_type = random.choice(['U', 'V', 'W'])
        if var_type == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, r - 1)
            old_val = U[row][col]
            new_val = random.choice([v for v in values if v != old_val])
            U[row][col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error > current_error:
                U[row][col] = old_val
            else:
                current_error = new_error
        elif var_type == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, r - 1)
            old_val = V[row][col]
            new_val = random.choice([v for v in values if v != old_val])
            V[row][col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error > current_error:
                V[row][col] = old_val
            else:
                current_error = new_error
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, r - 1)
            old_val = W[row][col]
            new_val = random.choice([v for v in values if v != old_val])
            W[row][col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error > current_error:
                W[row][col] = old_val
            else:
                current_error = new_error
        
        if current_error < 1e-12:
            break
    
    return U, V, W, current_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Schoolbook is a valid baseline
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank solution
    # For rank r, we have 45r variables and 50 equations
    # Try ranks from 25 down to 10
    values = [-1, -0.5, 0, 0.5, 1]
    
    for r in range(25, 9, -1):
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            U, V, W = _random_init(n, m, p, r, values)
            U, V, W, err = _local_search(U, V, W, n, m, p, r, values, max_iters=200, timeout=20)
            
            if err < 1e-9:
                # Verify exactly
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
                    break
        
        if best_rank <= r:
            break
    
    return best_result
