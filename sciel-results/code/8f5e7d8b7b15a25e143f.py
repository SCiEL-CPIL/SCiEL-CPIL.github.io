"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W of given rank satisfy the decomposition exactly."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def random_sparse_matrix(rows, cols, density=0.3, values=(-1, 1)):
        """Generate a random sparse matrix with given density and values."""
        mat = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                if random.random() < density:
                    mat[i, j] = random.choice(values)
        return mat
    
    def local_improve(U, V, W, rank, max_iters=5000):
        """Local improvement: try flipping individual entries to reduce error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        def compute_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        w_val = W[k * n + i]
                        total = np.dot(u_val, v_val * w_val)
                        err += (total - 1.0) ** 2
            return err
        
        current_error = compute_error()
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            if current_error < 1e-9:
                if check_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
                    return U, V, W
                break
            
            # Try flipping a random entry in one of the matrices
            mat_id = random.randint(0, 2)
            if mat_id == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row, col]
                if old_val == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = 0
                U[row, col] = new_val
                new_error = compute_error()
                if new_error >= current_error:
                    U[row, col] = old_val
                else:
                    current_error = new_error
            elif mat_id == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row, col]
                if old_val == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = 0
                V[row, col] = new_val
                new_error = compute_error()
                if new_error >= current_error:
                    V[row, col] = old_val
                else:
                    current_error = new_error
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row, col]
                if old_val == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = 0
                W[row, col] = new_val
                new_error = compute_error()
                if new_error >= current_error:
                    W[row, col] = old_val
                else:
                    current_error = new_error
        
        return None
    
    # Try ranks from 40 to 71
    for rank in range(40, 72):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_attempts = 30 if rank < 50 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization
            density = random.choice([0.2, 0.3, 0.4])
            U = random_sparse_matrix(n * m, rank, density)
            V = random_sparse_matrix(m * p, rank, density)
            W = random_sparse_matrix(p * n, rank, density)
            
            # Check if already exact
            if check_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
            
            # Try local improvement
            result = local_improve(U, V, W, rank)
            if result:
                U_int, V_int, W_int = result
                best_rank = rank
                best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
