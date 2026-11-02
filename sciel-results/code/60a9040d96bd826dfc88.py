import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

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

def _compute_error(U, V, W, T):
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.linalg.norm(reconstructed - T)

def _check_exact(U, V, W, n, m, p):
    T = _build_target_tensor(n, m, p)
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-9)

def _try_rank(R, n, m, p, T, time_limit):
    start = time.time()
    values = [-1, 0, 1]
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few random restarts
    max_restarts = 15
    for _ in range(max_restarts):
        if time.time() - start > time_limit:
            break
            
        U = np.random.choice(values, size=(n * m, R))
        V = np.random.choice(values, size=(m * p, R))
        W = np.random.choice(values, size=(p * n, R))
        
        err = _compute_error(U, V, W, T)
        
        # Local search with better move selection
        for _ in range(2000):
            if time.time() - start > time_limit:
                break
            if err == 0:
                break
            
            # Pick random variable
            total_vars = U.size + V.size + W.size
            idx = random.randint(0, total_vars - 1)
            
            if idx < U.size:
                mat = U
                local_idx = idx
            elif idx < U.size + V.size:
                mat = V
                local_idx = idx - U.size
            else:
                mat = W
                local_idx = idx - U.size - V.size
                
            row, col = np.unravel_index(local_idx, mat.shape)
            old_val = mat[row, col]
            
            # Try all possible values and pick the best
            best_new_val = old_val
            best_new_err = err
            
            for new_val in values:
                if new_val == old_val:
                    continue
                mat[row, col] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_new_val = new_val
            
            if best_new_val != old_val:
                mat[row, col] = best_new_val
                err = best_new_err
            else:
                mat[row, col] = old_val
                
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0
    
    # Search for lower ranks
    # Lower bound is 12
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
            
        time_budget = 8.0 # 8 seconds per rank
        result = _try_rank(R, n, m, p, T, time_budget)
        
        if result is not None:
            U, V, W = result
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
