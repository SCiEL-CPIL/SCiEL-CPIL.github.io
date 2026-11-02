"""Matrix multiplication 2x5x5 via stochastic local search for rank reduction."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c_idx])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _local_search(n, m, p, target_rank, time_limit=10):
    """
    Attempts to find a decomposition of rank target_rank using local search.
    Uses random initialization and hill climbing on the error.
    """
    start_time = time.time()
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    # Precompute tensor indices
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_indices.append((i * m + j, j * p + k, k * n + i))
                
    def calculate_error(U, V, W):
        error = 0.0
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for a, b, c in tensor_indices:
            val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
            error += (val - 1.0) ** 2
        return error

    # Initialize
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    # Try a few random restarts
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization with small integers
        U = np.random.randint(-1, 2, size=(n_rows_U, target_rank))
        V = np.random.randint(-1, 2, size=(n_rows_V, target_rank))
        W = np.random.randint(-1, 2, size=(n_rows_W, target_rank))
        
        # Scale to help convergence? 
        # For now, keep integers.
        
        current_error = calculate_error(U.tolist(), V.tolist(), W.tolist())
        
        # Local search
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Randomly pick a matrix and an entry to perturb
            mat_choice = np.random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                rows = n_rows_U
            elif mat_choice == 'V':
                mat = V
                rows = n_rows_V
            else:
                mat = W
                rows = n_rows_W
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, target_rank)
            
            # Try changing value to -1, 0, 1
            old_val = mat[row, col]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                mat[row, col] = new_val
                new_error = calculate_error(U.tolist(), V.tolist(), W.tolist())
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                    if current_error == 0:
                        return U.tolist(), V.tolist(), W.tolist()
                else:
                    mat[row, col] = old_val
                    
        if current_error < best_error:
            best_error = current_error
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank < 50 decomposition
    # Start from 49 down to 45
    for rank in range(n * m * p - 1, 44, -1):
        result = _local_search(n, m, p, rank, time_limit=5)
        if result:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
