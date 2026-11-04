import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Try ranks in increasing order of likelihood/effort
    # Lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # Directive suggests trying 18. We try it briefly.
    # Then 27, then 30, etc.
    
    candidates = [18, 27, 30, 33, 36, 40, 45, 50, 54]
    
    # Allocate time
    # 18: 2s
    # 27: 40s
    # 30: 20s
    # 33: 20s
    # Rest: 5s each
    time_alloc = {
        18: 2.0,
        27: 40.0,
        30: 20.0,
        33: 20.0
    }
    
    for R in candidates:
        if time.time() > deadline - 5:
            break
        t_budget = time_alloc.get(R, 5.0)
        # Ensure we don't exceed deadline
        if time.time() + t_budget > deadline - 2:
            t_budget = max(0, deadline - 2 - time.time())
        
        if t_budget <= 0:
            continue
            
        result = search_low_rank(n, m, p, R, t_budget)
        if result is not None:
            return result

    # Fallback to schoolbook
    return construct_schoolbook(n, m, p)

def search_low_rank(n, m, p, target_rank, time_budget):
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def calc_error(U, V, W):
        Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = Res - T
        return np.sum(diff**2)

    random.seed(42)
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start = time.time()
    deadline = start + time_budget
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize
        U = np.random.randint(-1, 2, size=(num_u, target_rank)).astype(float)
        V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)
        
        current_err = calc_error(U, V, W)
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                return snap_and_verify(best_U, best_V, best_W, calc_error)
        
        # Local Search
        deltas = [0.5, -0.5, 1.0, -1.0]
        no_improve_count = 0
        max_no_improve = 500 # Restart if stuck
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows = num_u
            elif mat_idx == 1:
                mat = V
                rows = num_v
            else:
                mat = W
                rows = num_w
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, target_rank - 1)
            
            old_val = mat[row, col]
            delta = random.choice(deltas)
            new_val = old_val + delta
            
            mat[row, col] = new_val
            new_err = calc_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                no_improve_count = 0
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return snap_and_verify(best_U, best_V, best_W, calc_error)
            else:
                mat[row, col] = old_val
                no_improve_count += 1
        
        # If we get here, we restart with new random init

    # Final check on best found
    if best_err < 1e-6:
        return snap_and_verify(best_U, best_V, best_W, calc_error)

    return None

def snap_and_verify(U, V, W, calc_error):
    bU = np.round(U * 2) / 2
    bV = np.round(V * 2) / 2
    bW = np.round(W * 2) / 2
    snap_err = calc_error(bU, bV, bW)
    if snap_err < 1e-9:
        return {
            "U": bU.tolist(),
            "V": bV.tolist(),
            "W": bW.tolist()
        }
    return None

def construct_schoolbook(n, m, p):
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
