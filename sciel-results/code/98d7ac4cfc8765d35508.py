import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Target tensor T
    # T is indexed by (i, j, k) for A_ij, B_jk, C_ik
    # Flattened indices:
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i  (Note: W index in seed is k*n+i)
    
    # We want to find U, V, W such that:
    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] == T[u_idx, v_idx, w_idx]
    
    # T is 1 if (u_idx, v_idx, w_idx) corresponds to a valid (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Result: (nm, mp, pn)
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a batched matrix multiplication or einsum
        # np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def error(U, V, W):
        T_calc = compute_tensor(U, V, W)
        return np.sum((T_calc - target) ** 2)

    def check_and_format(U, V, W):
        # Check if U, V, W are integer/half-integer
        # And if they exactly decompose
        # The prompt says "integer/half-integer entries"
        # We assume the search will find such values.
        # We should round to nearest 0.5 before final check to ensure format?
        # Or just check if current values are close to 0.5 multiples.
        
        # For the return value, we need lists of lists.
        # The prompt says "exactly decompose".
        # If we use floats, we might have precision issues.
        # We will enforce 0.5 multiples in the search.
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

    # Search for rank R
    # Known lower bounds:
    # M_2,3,4 rank >= 18?
    # Let's try 17, 18, 19, 20...
    
    best_result = None
    best_rank = n * m * p
    
    # Naive solution
    U_naive = np.zeros((n * m, best_rank), dtype=float)
    V_naive = np.zeros((m * p, best_rank), dtype=float)
    W_naive = np.zeros((p * n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1.0
                V_naive[j * p + k, col] = 1.0
                W_naive[k * n + i, col] = 1.0
                col += 1
    best_result = check_and_format(U_naive, V_naive, W_naive)
    
    for R in range(17, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small integers
            # Values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            W = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            # Scale? Sometimes larger values are needed.
            # But for rank 18, small integers are likely.
            
            err = error(U, V, W)
            
            # Local Search
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                if err == 0:
                    found = True
                    break
                    
                # Pick a random factor
                factor_idx = np.random.randint(3)
                if factor_idx == 0:
                    # Perturb U
                    row = np.random.randint(n * m)
                    col = np.random.randint(R)
                    # Try adding -1, 1, 0.5, -0.5
                    # To keep it simple, try integer steps first, then half-integer
                    # Or just try delta in [-1, 1]
                    # Let's try changing one element
                    old_val = U[row, col]
                    # Try a few random deltas
                    delta = np.random.choice([-1.0, 1.0, -0.5, 0.5])
                    new_val = old_val + delta
                    U[row, col] = new_val
                    new_err = error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        U[row, col] = old_val
                        
                elif factor_idx == 1:
                    # Perturb V
                    row = np.random.randint(m * p)
                    col = np.random.randint(R)
                    old_val = V[row, col]
                    delta = np.random.choice([-1.0, 1.0, -0.5, 0.5])
                    new_val = old_val + delta
                    V[row, col] = new_val
                    new_err = error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        V[row, col] = old_val
                else:
                    # Perturb W
                    row = np.random.randint(p * n)
                    col = np.random.randint(R)
                    old_val = W[row, col]
                    delta = np.random.choice([-1.0, 1.0, -0.5, 0.5])
                    new_val = old_val + delta
                    W[row, col] = new_val
                    new_err = error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        W[row, col] = old_val
            
            if found:
                # Verify exactly
                # The values might be floats like 0.999999
                # We should snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if error(U_snap, V_snap, W_snap) == 0:
                    best_result = check_and_format(U_snap, V_snap, W_snap)
                    best_rank = R
                    break # Found minimal rank for this R, break to next R?
                    # Actually, we want the MINIMAL rank.
                    # So if we find R=17, we should stop.
                    # The loop goes 17, 18... so if we find 17, we break the R loop.
                    break
        
        if found:
            break
            
    return best_result
