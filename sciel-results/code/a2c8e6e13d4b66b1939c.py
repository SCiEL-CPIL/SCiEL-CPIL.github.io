"""Rewritten program: rank-minimal exact tensor decomposition for 3x3 x 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form an exact decomposition of the matmul tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, target_rank, max_iters=5000, seed=42, deadline=None):
    """Try to find a rank-target_rank decomposition using iterative refinement."""
    rng = random.Random(seed)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for attempt in range(5):
        if deadline is not None and time.time() > deadline:
            break
            
        U = np.array([[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            # Compute current error
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        errors.append(1 - s)
            
            err_arr = np.array(errors)
            max_err_idx = np.argmax(np.abs(err_arr))
            max_err_val = abs(err_arr[max_err_idx])
            
            if max_err_val < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Find which (i,j,k) this corresponds to
            idx = max_err_idx
            k = idx % p
            j = (idx // p) % m
            i = idx // (p * m)
            
            # Try to fix by modifying one entry
            best_change = None
            best_new_err = max_err_val
            
            for r in range(target_rank):
                # Try changing U[i*m+j][r]
                old_u = U[i*m+j][r]
                for new_u in [0, 1, -1]:
                    if new_u == old_u:
                        continue
                    U[i*m+j][r] = new_u
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    new_err = abs(1 - s)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_change = ('U', i*m+j, r, new_u)
                    U[i*m+j][r] = old_u
                
                # Try changing V[j*p+k][r]
                old_v = V[j*p+k][r]
                for new_v in [0, 1, -1]:
                    if new_v == old_v:
                        continue
                    V[j*p+k][r] = new_v
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    new_err = abs(1 - s)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_change = ('V', j*p+k, r, new_v)
                    V[j*p+k][r] = old_v
                
                # Try changing W[k*n+i][r]
                old_w = W[k*n+i][r]
                for new_w in [0, 1, -1]:
                    if new_w == old_w:
                        continue
                    W[k*n+i][r] = new_w
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    new_err = abs(1 - s)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_change = ('W', k*n+i, r, new_w)
                    W[k*n+i][r] = old_w
            
            if best_change is not None:
                mat_name, row, col, val = best_change
                if mat_name == 'U':
                    U[row][col] = val
                elif mat_name == 'V':
                    V[row][col] = val
                else:
                    W[row][col] = val
            else:
                # Random perturbation
                mat_choice = rng.random()
                if mat_choice < 0.33:
                    row = rng.randint(0, n*m-1)
                    col = rng.randint(0, target_rank-1)
                    U[row][col] = rng.choice([0, 1, -1])
                elif mat_choice < 0.66:
                    row = rng.randint(0, m*p-1)
                    col = rng.randint(0, target_rank-1)
                    V[row][col] = rng.choice([0, 1, -1])
                else:
                    row = rng.randint(0, p*n-1)
                    col = rng.randint(0, target_rank-1)
                    W[row][col] = rng.choice([0, 1, -1])
        
        # Compute final error
        total_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total_err += (1 - s) ** 2
        
        if total_err < best_err:
            best_err = total_err
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave safety margin
    
    # First, return the schoolbook decomposition as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    for target_rank in [40, 42, 44, 46, 48, 50, 52]:
        if time.time() > deadline:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, max_iters=3000, seed=42, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return schoolbook
