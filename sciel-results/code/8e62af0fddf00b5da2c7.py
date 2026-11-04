import numpy as np
import time
import random
import sys

DIMENSIONS = [4, 5, 6]

def _make_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    
    # Check tensor identity using einsum
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _rkk_search(n, m, p, R, time_limit, seed):
    """
    Search for a rank-R decomposition using discrete randomization and Karmarkar-Karp inspired hashing.
    Entries are restricted to integers {-1, 0, 1} to facilitate exact cancellation checks.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize with random sparse integer matrices
    # Sparsity helps in finding cancellations
    def rand_sparse_int(shape, density=0.1):
        mat = np.zeros(shape, dtype=int)
        idx = np.random.choice(shape[0] * shape[1], int(shape[0] * shape[1] * density), replace=False)
        rows, cols = np.unravel_index(idx, shape)
        vals = np.random.choice([-1, 1], len(idx))
        mat[rows, cols] = vals
        return mat

    U = rand_sparse_int((n * m, R), 0.15)
    V = rand_sparse_int((m * p, R), 0.15)
    W = rand_sparse_int((p * n, R), 0.15)
    
    # Precompute reshapes
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # Let's compute the current tensor T and error
    # To speed up, we can track errors per (i,j,k)
    
    # Optimization: Instead of full einsum every step, we can do local updates.
    # But for simplicity and robustness in 150s, we'll do a mix of random flips and local corrections.
    
    # Hashing trick: For a fixed (i,j), the contribution to T[i,j,k] is sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]).
    # This is still complex. Let's stick to a simpler discrete annealing.
    
    start_time = time.time()
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Local error calculation
    def calc_err(U3, V3, W3):
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.sum((T - 1.0)**2), T

    err, T = calc_err(U3, V3, W3)
    if err == 0:
        return U.astype(float), V.astype(float), W.astype(float)
    best_err = err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_T = T.copy()
    
    # Simple simulated annealing on integer flips
    T_cur_temp = 1.0
    T_min_temp = 0.01
    cooling_rate = 0.95
    
    while time.time() - start_time < time_limit:
        # Pick a random matrix and entry to flip
        mat_idx = random.choice([0, 1, 2])
        if mat_idx == 0:
            r, c = random.randint(0, n*m-1), random.randint(0, R-1)
            if U[r, c] == 0: continue # Flipping 0 to +/-1 might be too aggressive, try flipping non-zero first
            # Actually, let's allow flipping 0 to 1 or -1, and 1 to 0 or -1
            old_val = U[r, c]
            if old_val == 0:
                new_val = random.choice([-1, 1])
            else:
                new_val = random.choice([0, -old_val])
            
            # Calculate delta in T
            # T[i,j,k] changes by delta_U * V[j,k,c] * W[k,i,c]
            i, j = divmod(r, m)
            delta_U = new_val - old_val
            if delta_U == 0: continue
            
            # V3[j, :, c] is shape (p,)
            # W3[:, i, c] is shape (p,)
            # Contribution to T[i, j, k] is delta_U * V[j, k, c] * W[k, i, c]
            
            v_vec = V3[j, :, c] # shape (p,)
            w_vec = W3[:, i, c] # shape (p,)
            
            # T[i, j, k] += delta_U * v_vec[k] * w_vec[k]
            delta_T = delta_U * (v_vec * w_vec) # shape (p,)
            
            old_T_slice = best_T[i, j, :]
            new_T_slice = old_T_slice + delta_T
            
            old_err_slice = np.sum((old_T_slice - 1.0)**2)
            new_err_slice = np.sum((new_T_slice - 1.0)**2)
            delta_err = new_err_slice - old_err_slice
            
            if delta_err < 0 or (T_cur_temp > 0.01 and random.random() < np.exp(-delta_err / T_cur_temp)):
                U[r, c] = new_val
                U3[i, j, c] = new_val
                best_T[i, j, :] = new_T_slice
                best_err += delta_err
                
                if best_err < 1e-6:
                    # Verify and snap
                    U_f = U.astype(float)
                    V_f = V.astype(float)
                    W_f = W.astype(float)
                    if _verify_fast(U_f, V_f, W_f):
                        return U_f, V_f, W_f
                
                if best_err < best_err: # Update best if improved (always true here)
                    pass # We are tracking current state as best in this simplified SA
            else:
                # Reject move
                pass
                
        elif mat_idx == 1:
            r, c = random.randint(0, m*p-1), random.randint(0, R-1)
            if V[r, c] == 0: continue
            old_val = V[r, c]
            if old_val == 0:
                new_val = random.choice([-1, 1])
            else:
                new_val = random.choice([0, -old_val])
            
            j, k = divmod(r, p)
            delta_V = new_val - old_val
            if delta_V == 0: continue
            
            u_vec = U3[:, j, c] # shape (n,)
            w_vec = W3[k, :, c] # shape (n,)
            
            # T[i, j, k] += u_vec[i] * delta_V * w_vec[i]
            delta_T = delta_V * (u_vec * w_vec) # shape (n,)
            
            old_T_slice = best_T[:, j, k]
            new_T_slice = old_T_slice + delta_T
            
            old_err_slice = np.sum((old_T_slice - 1.0)**2)
            new_err_slice = np.sum((new_T_slice - 1.0)**2)
            delta_err = new_err_slice - old_err_slice
            
            if delta_err < 0 or (T_cur_temp > 0.01 and random.random() < np.exp(-delta_err / T_cur_temp)):
                V[r, c] = new_val
                V3[j, k, c] = new_val
                best_T[:, j, k] = new_T_slice
                best_err += delta_err
                
                if best_err < 1e-6:
                    U_f = U.astype(float)
                    V_f = V.astype(float)
                    W_f = W.astype(float)
                    if _verify_fast(U_f, V_f, W_f):
                        return U_f, V_f, W_f

        else:
            r, c = random.randint(0, p*n-1), random.randint(0, R-1)
            if W[r, c] == 0: continue
            old_val = W[r, c]
            if old_val == 0:
                new_val = random.choice([-1, 1])
            else:
                new_val = random.choice([0, -old_val])
            
            k, i = divmod(r, n)
            delta_W = new_val - old_val
            if delta_W == 0: continue
            
            u_vec = U3[:, :, c] # shape (n*m,) -> need to reshape for index i
            # U3 is (n, m, R). We need U[i, j, c] for all j.
            u_row = U3[i, :, c] # shape (m,)
            v_vec = V3[:, k, c] # shape (m,)
            
            # T[i, j, k] += u_row[j] * v_vec[j] * delta_W
            delta_T = delta_W * (u_row * v_vec) # shape (m,)
            
            old_T_slice = best_T[i, :, k]
            new_T_slice = old_T_slice + delta_T
            
            old_err_slice = np.sum((old_T_slice - 1.0)**2)
            new_err_slice = np.sum((new_T_slice - 1.0)**2)
            delta_err = new_err_slice - old_err_slice
            
            if delta_err < 0 or (T_cur_temp > 0.01 and random.random() < np.exp(-delta_err / T_cur_temp)):
                W[r, c] = new_val
                W3[k, i, c] = new_val
                best_T[i, :, k] = new_T_slice
                best_err += delta_err
                
                if best_err < 1e-6:
                    U_f = U.astype(float)
                    V_f = V.astype(float)
                    W_f = W.astype(float)
                    if _verify_fast(U_f, V_f, W_f):
                        return U_f, V_f, W_f

        # Cooling
        if T_cur_temp > T_min_temp:
            T_cur_temp *= cooling_rate
        else:
            # Restart with higher temp if stuck
            T_cur_temp = 0.5
            # Random restart of a few entries
            for _ in range(10):
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    r, c = random.randint(0, n*m-1), random.randint(0, R-1)
                    U[r, c] = random.choice([-1, 0, 1])
                    U3[divmod(r, m)[0], divmod(r, m)[1], c] = U[r, c]
                elif mat_idx == 1:
                    r, c = random.randint(0, m*p-1), random.randint(0, R-1)
                    V[r, c] = random.choice([-1, 0, 1])
                    V3[divmod(r, p)[0], divmod(r, p)[1], c] = V[r, c]
                else:
                    r, c = random.randint(0, p*n-1), random.randint(0, R-1)
                    W[r, c] = random.choice([-1, 0, 1])
                    W3[divmod(r, n)[0], divmod(r, n)[1], c] = W[r, c]
            
            # Recalculate full error to resync
            err, T = calc_err(U3, V3, W3)
            best_err = err
            best_T = T.copy()

    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # 2. Try discrete search for lower ranks
    # Schoolbook rank is 120.
    # Lower bounds are around 100-110 for 4x5x6?
    # Let's try a few ranks below 120.
    
    seeds = [42, 123, 999, 777, 1, 2, 3, 4, 5]
    
    # Try ranks from 119 down to 105
    # Time budget per rank
    ranks_to_try = range(119, 104, -1)
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        
        # Allocate time per rank
        # If time is tight, fewer restarts
        max_restarts = 3 if time_left < 30 else 5
        
        for seed_idx in range(max_restarts):
            if time.time() > deadline:
                break
                
            seed = seeds[seed_idx % len(seeds)] + R
            
            # Allocate time per restart
            # If we have 100s left and 5 restarts, 20s each
            # But we want to check earlier.
            restart_time_limit = min(10.0, time_left / (max_restarts - seed_idx))
            
            U_s, V_s, W_s = _rkk_search(n, m, p, R, restart_time_limit, seed)
            
            if U_s is not None:
                # Final verification
                if _verify_fast(U_s, V_s, W_s):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    return best_result

# Fallback in case construct is not called or for direct testing
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = DIMENSIONS
    ok = _verify_fast(U, V, W)
    if ok:
        print("Verification Passed")
    else:
        print("Verification Failed")
