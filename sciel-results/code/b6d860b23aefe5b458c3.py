import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]
VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                nz_indices.append((u, v, w))
    return T, nz_indices

def _compute_error(U, V, W, T, nz_indices):
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.sum((recon - T) ** 2)

def _check_exact(U, V, W, T, nz_indices, n, m, p):
    # Check non-zero positions
    for u, v, w in nz_indices:
        val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
        if abs(val - 1.0) > 1e-9:
            return False
    
    # Check zero positions
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) in nz_indices:
                    continue
                val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                if abs(val) > 1e-9:
                    return False
    return True

def _get_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _search_rank(R, n, m, p, T, nz_indices, time_limit):
    start_time = time.time()
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Use more restarts and longer runs if time permits
    # Estimate time per restart
    num_restarts = 200
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small integers
        U = np.random.choice(VALUES, size=(n * m, R))
        V = np.random.choice(VALUES, size=(m * p, R))
        W = np.random.choice(VALUES, size=(p * n, R))
        
        current_error = _compute_error(U, V, W, T, nz_indices)
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if best_error < 1e-9:
                return best_U, best_V, best_W
                
        # Local Search: Simulated Annealing
        max_iter = 2000
        temp = 1.0
        cooling = 0.995
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if current_error < 1e-9:
                break
                
            # Pick a move
            # 1. Change one entry
            # 2. Change a random column entry in U, V, or W
            move_type = random.random()
            
            if move_type < 0.7:
                # Change one entry
                total_vars = U.size + V.size + W.size
                var_idx = random.randint(0, total_vars - 1)
                
                if var_idx < U.size:
                    mat = U
                    idx = var_idx
                elif var_idx < U.size + V.size:
                    mat = V
                    idx = var_idx - U.size
                else:
                    mat = W
                    idx = var_idx - U.size - V.size
                    
                old_val = mat.flat[idx]
                new_val = random.choice(VALUES)
                if new_val == old_val:
                    continue
                    
                mat.flat[idx] = new_val
                new_error = _compute_error(U, V, W, T, nz_indices)
                
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat.flat[idx] = old_val
            else:
                # Change a random entry in a specific matrix, but try to keep column structure?
                # Just change 2 entries
                total_vars = U.size + V.size + W.size
                var1 = random.randint(0, total_vars - 1)
                var2 = random.randint(0, total_vars - 1)
                if var1 == var2:
                    continue
                
                def get_mat_idx(idx):
                    if idx < U.size:
                        return U, idx
                    elif idx < U.size + V.size:
                        return V, idx - U.size
                    else:
                        return W, idx - U.size - V.size
                
                mat1, idx1 = get_mat_idx(var1)
                mat2, idx2 = get_mat_idx(var2)
                
                old_val1 = mat1.flat[idx1]
                old_val2 = mat2.flat[idx2]
                
                new_val1 = random.choice(VALUES)
                new_val2 = random.choice(VALUES)
                
                if new_val1 == old_val1 and new_val2 == old_val2:
                    continue
                    
                mat1.flat[idx1] = new_val1
                mat2.flat[idx2] = new_val2
                
                new_error = _compute_error(U, V, W, T, nz_indices)
                
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat1.flat[idx1] = old_val1
                    mat2.flat[idx2] = old_val2
                    
            temp *= cooling
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T, nz_indices = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Try ranks from low to high
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
            
        # Allocate more time to lower ranks?
        # If we find a solution at R, we stop.
        # So we should spend time on the lowest R first.
        time_budget = 10.0
        if R == 12:
            time_budget = 20.0 # Give more time to the hardest rank
            
        result = _search_rank(R, n, m, p, T, nz_indices, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify exactness again just in case
            if _check_exact(U, V, W, T, nz_indices, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
