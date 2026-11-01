import numpy as np
import time

def _search_2x3x3_rank_15():
    # Tensor for 2x3 x 3x3
    # U: 6x15, V: 9x15, W: 6x15
    # T[u, v, w] = 1 if u=i*3+j, v=j*3+k, w=k*2+i
    T = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u = i * 3 + j
                v = j * 3 + k
                w = k * 2 + i
                T[u, v, w] = 1.0
                
    rng = np.random.default_rng(123)
    alphabet = np.array([-1, 0, 1])
    
    # Initial random guess
    U = rng.choice(alphabet, size=(6, 15))
    V = rng.choice(alphabet, size=(9, 15))
    W = rng.choice(alphabet, size=(6, 15))
    
    def calc_error(U, V, W):
        # Compute S and compare with T
        # S = sum_r U[:,r][:,None,None] * V[:,r][None,:,None] * W[:,r][None,None,:]
        # To speed up, we can use einsum or loops
        S = np.zeros((6, 9, 6))
        for r in range(15):
            S += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        # Error is number of mismatches
        # T has 1s at specific places, 0s elsewhere.
        # We want S to match T.
        # Since entries are integers, we can check exact match.
        # However, random search might produce large numbers.
        # We minimize sum of squared differences?
        # For now, count mismatches.
        return np.sum(S != T)
    
    current_err = calc_error(U, V, W)
    
    # Local search
    # Change one entry in U, V, or W
    # If error decreases, accept.
    
    max_iters = 5000
    for _ in range(max_iters):
        # Pick a random matrix and index
        mat_choice = rng.integers(0, 3)
        if mat_choice == 0:
            idx = (rng.integers(0, 6), rng.integers(0, 15))
            old_val = U[idx]
            new_val = rng.choice(alphabet)
            if old_val == new_val: continue
            U[idx] = new_val
            new_err = calc_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
            else:
                U[idx] = old_val
        elif mat_choice == 1:
            idx = (rng.integers(0, 9), rng.integers(0, 15))
            old_val = V[idx]
            new_val = rng.choice(alphabet)
            if old_val == new_val: continue
            V[idx] = new_val
            new_err = calc_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
            else:
                V[idx] = old_val
        else:
            idx = (rng.integers(0, 6), rng.integers(0, 15))
            old_val = W[idx]
            new_val = rng.choice(alphabet)
            if old_val == new_val: continue
            W[idx] = new_val
            new_err = calc_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
            else:
                W[idx] = old_val
                
        if current_err == 0:
            return U, V, W
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find 2x3x3 rank 15
    decomp_233 = _search_2x3x3_rank_15()
    
    if decomp_233:
        U0, V0, W0 = decomp_233
        # Construct 2x3x9 from 2x3x3
        # Rank 45
        U = np.zeros((6, 45))
        V = np.zeros((27, 45))
        W = np.zeros((18, 45))
        
        for k_prime in range(3):
            start = k_prime * 15
            end = start + 15
            
            U[:, start:end] = U0
            
            # V placement
            # V_full index: j*9 + k
            # V0 index: j*3 + k''
            # k = 3*k_prime + k''
            # Row in V_full: j*9 + 3*k_prime + k''
            # Row in V0: j*3 + k''
            for j in range(3):
                for k_prime_prime in range(3):
                    row_v0 = j * 3 + k_prime_prime
                    row_v_full = j * 9 + 3 * k_prime + k_prime_prime
                    V[row_v_full, start:end] = V0[row_v0, :]
                    
            # W placement
            # W_full index: k*2 + i
            # W0 index: k''*2 + i
            # k = 3*k_prime + k''
            # Row in W_full: (3*k_prime + k'')*2 + i = 6*k_prime + 2*k'' + i
            # Row in W0: k''*2 + i
            for k_prime_prime in range(3):
                for i in range(2):
                    row_w0 = k_prime_prime * 2 + i
                    row_w_full = 6 * k_prime + 2 * k_prime_prime + i
                    W[row_w_full, start:end] = W0[row_w0, :]
                    
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        n, m, p = 2, 3, 9
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
