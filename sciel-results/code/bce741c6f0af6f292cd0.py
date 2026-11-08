"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using randomized local search
    # We search for solutions with entries in small integer sets
    
    # Strategy: For a given rank r, try to find U, V, W such that
    # for all (i,j,k): sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Use a greedy + local search approach
    
    for target_rank in range(70, 55, -1):
        if time.time() > deadline - 15:
            break
        
        found = _find_decomposition(n, m, p, target_rank, deadline - 15)
        if found is not None:
            U, V, W = found
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}


def _find_decomposition(n, m, p, rank, time_limit):
    """Try to find a rank decomposition using randomized local search."""
    deadline = time.time() + time_limit
    random.seed(42)  # for reproducibility
    
    # We need to satisfy: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_{t=0}^{rank-1} U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # Use a greedy initialization followed by local search
    
    # Initialize with random sparse values
    max_val = 2  # entries in {-2, -1, 0, 1, 2}
    
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Randomly assign some non-zero values
        for t in range(rank):
            # Each column t gets some non-zero entries
            num_nonzero_u = random.randint(2, 5)
            num_nonzero_v = random.randint(2, 5)
            num_nonzero_w = random.randint(2, 5)
            
            u_idx = random.sample(range(n * m), num_nonzero_u)
            v_idx = random.sample(range(m * p), num_nonzero_v)
            w_idx = random.sample(range(p * n), num_nonzero_w)
            
            for idx in u_idx:
                U[idx, t] = random.choice([-2, -1, 1, 2])
            for idx in v_idx:
                V[idx, t] = random.choice([-2, -1, 1, 2])
            for idx in w_idx:
                W[idx, t] = random.choice([-2, -1, 1, 2])
        
        # Compute current error
        error, satisfied = _compute_error(n, m, p, U, V, W)
        
        if error == 0:
            # Verify and return
            if _verify(n, m, p, U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        # Local search: try to improve by modifying individual entries
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try random modifications
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Choose which matrix to modify
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, rank - 1)
                    old_val = U[row, col]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    U[row, col] = new_val
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    if error_new < error:
                        error = error_new
                        improved = True
                        if error == 0 and _verify(n, m, p, U, V, W):
                            return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        U[row, col] = old_val
                
                elif mat == 'V':
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, rank - 1)
                    old_val = V[row, col]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    V[row, col] = new_val
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    if error_new < error:
                        error = error_new
                        improved = True
                        if error == 0 and _verify(n, m, p, U, V, W):
                            return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        V[row, col] = old_val
                
                else:  # W
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, rank - 1)
                    old_val = W[row, col]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    W[row, col] = new_val
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    if error_new < error:
                        error = error_new
                        improved = True
                        if error == 0 and _verify(n, m, p, U, V, W):
                            return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        W[row, col] = old_val
        
        # Track best
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    # If we found a solution with small error but not zero, try to polish
    if best_U is not None and best_error <= 5:
        # Try a more focused search
        if time.time() < deadline - 5:
            polished = _polish_solution(n, m, p, best_U, best_V, best_W, deadline - 5)
            if polished is not None:
                return polished
    
    return None


def _compute_error(n, m, p, U, V, W):
    """Compute the error (number of unsatisfied constraints) and count satisfied."""
    rank = U.shape[1]
    error = 0
    satisfied = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                val = 0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if val == 1:
                    satisfied += 1
                else:
                    error += 1
    return error, satisfied


def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if val != 1:
                    return False
    return True


def _polish_solution(n, m, p, U, V, W, time_limit):
    """Try to polish a near-solution to an exact solution."""
    deadline = time.time() + time_limit
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Find unsatisfied constraints
    unsatisfied = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if val != 1:
                    unsatisfied.append((i, j, k, val))
    
    if not unsatisfied:
        return (U.tolist(), V.tolist(), W.tolist())
    
    # Try to fix each unsatisfied constraint
    for i, j, k, current_val in unsatisfied:
        if time.time() > deadline:
            break
        
        target = 1
        diff = target - current_val
        
        # Try to adjust one entry to fix this constraint
        # U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should change by diff
        
        # Try modifying U[i*m+j,t] for various t
        for t in range(rank):
            v_val = V[j * p + k, t]
            w_val = W[k * n + i, t]
            if v_val == 0 or w_val == 0:
                continue
            product_vw = v_val * w_val
            if product_vw == 0:
                continue
            # We need U[i*m+j,t] * product_vw to change by diff
            # New U value: (current_val - old_U*product_vw + new_U*product_vw) should = target
            # So new_U = (target - (current_val - old_U*product_vw)) / product_vw
            old_u = U[i * m + j, t]
            residual = current_val - old_u * product_vw
            new_u_num = target - residual
            if new_u_num % product_vw == 0:
                new_u = new_u_num // product_vw
                if -2 <= new_u <= 2 and new_u != old_u:
                    # Check if this doesn't break other constraints too badly
                    U[i * m + j, t] = new_u
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    error_old, _ = _compute_error(n, m, p, U, V, W)
                    # Actually we just changed it, so error_old is after change
                    # Let's just accept if it reduces total error
                    if error_new <= len(unsatisfied):  # heuristic
                        if error_new == 0:
                            return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        U[i * m + j, t] = old_u
        
        # Try modifying V[j*p+k,t]
        for t in range(rank):
            u_val = U[i * m + j, t]
            w_val = W[k * n + i, t]
            if u_val == 0 or w_val == 0:
                continue
            product_uw = u_val * w_val
            if product_uw == 0:
                continue
            old_v = V[j * p + k, t]
            residual = current_val - old_v * product_uw
            new_v_num = target - residual
            if new_v_num % product_uw == 0:
                new_v = new_v_num // product_uw
                if -2 <= new_v <= 2 and new_v != old_v:
                    V[j * p + k, t] = new_v
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    if error_new == 0:
                        return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        V[j * p + k, t] = old_v
        
        # Try modifying W[k*n+i,t]
        for t in range(rank):
            u_val = U[i * m + j, t]
            v_val = V[j * p + k, t]
            if u_val == 0 or v_val == 0:
                continue
            product_uv = u_val * v_val
            if product_uv == 0:
                continue
            old_w = W[k * n + i, t]
            residual = current_val - old_w * product_uv
            new_w_num = target - residual
            if new_w_num % product_uv == 0:
                new_w = new_w_num // product_uv
                if -2 <= new_w <= 2 and new_w != old_w:
                    W[k * n + i, t] = new_w
                    error_new, _ = _compute_error(n, m, p, U, V, W)
                    if error_new == 0:
                        return (U.tolist(), V.tolist(), W.tolist())
                    else:
                        W[k * n + i, t] = old_w
    
    return None

# EVOLVE-BLOCK-END
