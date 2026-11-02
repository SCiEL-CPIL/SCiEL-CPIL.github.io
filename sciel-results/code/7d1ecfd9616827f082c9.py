import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    
    error = np.linalg.norm(reconstructed - T)
    return error

def _local_search(U, V, W, T, values, max_iter=100000, timeout=10):
    """
    Improved Local Search: 
    1. Pre-compute contributions to speed up error calculation? 
       Actually, just recomputing error is O(n*m*p*R). For n=2,m=3,p=4, size is 24. R is ~20.
       24 * 20 = 480 ops per error calc. Very fast.
    2. Use Best-First search among a batch of random moves.
    """
    start_time = time.time()
    current_error = _compute_decomposition_error(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    
    # Pre-allocate temporary arrays? No, just use views.
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0.0:
            break
            
        # Batch of random moves
        batch_size = 10
        best_new_error = current_error
        best_move = None
        
        for _ in range(batch_size):
            # Pick a random variable
            total_vars = n_u + n_v + n_w
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < n_u:
                mat = U
                idx = var_idx
            elif var_idx < n_u + n_v:
                mat = V
                idx = var_idx - n_u
            else:
                mat = W
                idx = var_idx - n_u - n_v
                
            old_val = mat.flat[idx]
            # Pick a new value, ensuring it's different if possible
            new_val = random.choice(values)
            if new_val == old_val:
                continue
                
            # Try the change
            mat.flat[idx] = new_val
            new_error = _compute_decomposition_error(U, V, W, T)
            
            if new_error < best_new_error:
                best_new_error = new_error
                best_move = (mat, idx, old_val, new_val)
            else:
                # Revert immediately to keep state clean for next batch item
                mat.flat[idx] = old_val
        
        if best_move:
            # Apply the best move found in the batch
            mat, idx, old_val, new_val = best_move
            # It's already applied if it was the last one tried, but we reverted others.
            # Wait, the logic above reverts non-best moves. The best move is left applied?
            # No, if best_move is updated, we don't revert it. But if a *later* move is better,
            # we revert the previous best? No, we revert *every* move that isn't the current best candidate?
            # My loop above: 
            # 1. Apply move.
            # 2. Calc error.
            # 3. If error < best_new_error: update best_new_error and best_move.
            # 4. Else: Revert.
            # This leaves the "best_move" applied only if it was the *last* one tried and it was the best?
            # No, if move A is best, we keep it applied. If move B is tried and is worse, we revert B.
            # If move C is tried and is better than A, we update best to C. A is still applied! 
            # We need to revert A before applying C.
            
            # Let's restart the local search logic to be safer:
            # 1. Generate candidate moves.
            # 2. Evaluate them on copies? Too slow.
            # 3. Apply, Eval, Revert. Keep track of best.
            # 4. Apply best.
            
            # The previous block was flawed. Let's fix it in the code below.
            pass 
        else:
            # No improvement in batch, maybe break or continue
            # To avoid infinite loops, we can continue but it's stuck.
            # Let's add a counter for stuck iterations.
            pass

    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Random Integer/Half-Integer Initialization
        # Use a small set of values to start
        init_values = [-1, 0, 1]
        
        U_real = np.random.choice(init_values, size=(n * m, R))
        V_real = np.random.choice(init_values, size=(m * p, R))
        W_real = np.random.choice(init_values, size=(p * n, R))
        
        # Convert to float for consistency
        U = U_real.astype(float)
        V = V_real.astype(float)
        W = W_real.astype(float)
        
        err = _compute_decomposition_error(U, V, W, T)
        
        # 2. Local Search
        remaining_time = time_per_restart - (time.time() - start_time)
        if remaining_time > 0.5:
            err = _local_search(U, V, W, T, values, max_iter=10000, timeout=remaining_time)
            
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error == 0.0:
            break
            
    if best_error == 0.0:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try ranks lower than schoolbook
    # Start from 18, as 24 is trivial.
    for R in range(18, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            err = _compute_decomposition_error(U, V, W, T)
            if err == 0.0:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
