"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    def compute_error(U, V, W, n, m, p):
        """Compute max error of decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try to find lower rank decompositions using optimization
    # Lower bound: max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    
    # Try ranks from a reasonable lower bound
    # Use a smooth loss: sum of squared errors
    def smooth_loss(U, V, W, n, m, p):
        rank = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    total_err += (val - 1.0) ** 2
        return total_err
    
    # Try to reduce rank from schoolbook
    # Strategy: start with schoolbook, then try to merge columns or find lower rank
    
    # For now, try a few target ranks with random restarts and gradient descent
    target_ranks = [30, 35, 40, 50, 60, 80, 100]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(12345 + restart + target_rank)
            
            # Initialize with small random values
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                loss = smooth_loss(U_try, V_try, W_try, n, m, p)
                if loss < 1e-10:
                    # Found valid decomposition
                    best_U = U_try.copy()
                    best_V = V_try.copy()
                    best_W = W_try.copy()
                    best_rank = target_rank
                    break
                
                # Compute gradients numerically (expensive but simple)
                # For efficiency, only do this if loss is decreasing
                if step % 10 == 0:
                    # Check if we're making progress
                    pass
                
                # Simple random perturbation instead of full gradient
                step_size = 0.01
                U_try += np.random.randn(n*m, target_rank) * step_size
                V_try += np.random.randn(m*p, target_rank) * step_size
                W_try += np.random.randn(p*n, target_rank) * step_size
            
            if best_rank == target_rank:
                break
    
    # Convert to list of lists with rounding
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Clean up near-zero and near-integer/half-integer values
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(best_U))
    V_out = clean_list(to_list(best_V))
    W_out = clean_list(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
