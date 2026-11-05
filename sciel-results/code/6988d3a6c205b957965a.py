"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Python integers to avoid floating point issues.
    """
    rank = len(U[0])
    
    T = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    for r in range(rank):
        u_col = [U[i * m + j][r] for i in range(n) for j in range(m)]
        v_col = [V[j * p + k][r] for j in range(m) for k in range(p)]
        w_col = [W[k * n + i][r] for k in range(p) for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    T[i][j][k] += u_val * v_val * w_val
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k] != 1:
                    return False
    return True

def _local_search(U, V, W, n, m, p, max_iters=2000, deadline=None):
    """
    Fast local search to minimize squared error.
    U, V, W are lists of lists (integers).
    Returns (U, V, W, error)
    """
    rank = len(U[0])
    
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    T = np.zeros((n, m, p), dtype=np.int64)
    
    # Initialize T
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                v_slice = v_col[j * p : (j + 1) * p]
                w_indices = [k * n + i for k in range(p)]
                w_vals = w_col[w_indices]
                T[i, j, :] += u_val * (v_slice * w_vals)
                
    error = np.sum((T - 1)**2)
    
    if error == 0:
        return U, V, W, 0

    for step in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        mat = random.randint(0, 2)
        r = random.randint(0, rank - 1)
        
        if mat == 0:
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            idx_u = i * m + j
            delta = random.choice([-1, 1])
            
            v_slice = V_np[j * p : (j + 1) * p, r]
            w_vals = W_np[[k * n + i for k in range(p)], r]
            change_vec = delta * (v_slice * w_vals)
            
            old_slice = T[i, j, :]
            new_slice = old_slice + change_vec
            
            old_err = np.sum((old_slice - 1)**2)
            new_err = np.sum((new_slice - 1)**2)
            delta_err = new_err - old_err
            
            if delta_err < 0:
                T[i, j, :] = new_slice
                U_np[idx_u, r] += delta
                error += delta_err
                if error == 0:
                    return U_np.tolist(), V_np.tolist(), W_np.tolist(), 0
        elif mat == 1:
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            idx_v = j * p + k
            delta = random.choice([-1, 1])
            
            u_vals = U_np[[i * m + j for i in range(n)], r]
            w_vals = W_np[k * n : (k + 1) * n, r]
            
            change_vec = delta * (u_vals * w_vals)
            
            old_slice = T[:, j, k]
            new_slice = old_slice + change_vec
            
            old_err = np.sum((old_slice - 1)**2)
            new_err = np.sum((new_slice - 1)**2)
            delta_err = new_err - old_err
            
            if delta_err < 0:
                T[:, j, k] = new_slice
                V_np[idx_v, r] += delta
                error += delta_err
                if error == 0:
                    return U_np.tolist(), V_np.tolist(), W_np.tolist(), 0
        else:
            k = random.randint(0, p - 1)
            i = random.randint(0, n - 1)
            idx_w = k * n + i
            delta = random.choice([-1, 1])
            
            u_vals = U_np[i * m : (i + 1) * m, r]
            v_vals = V_np[[j * p + k for j in range(m)], r]
            
            change_vec = delta * (u_vals * v_vals)
            
            old_slice = T[i, :, k]
            new_slice = old_slice + change_vec
            
            old_err = np.sum((old_slice - 1)**2)
            new_err = np.sum((new_slice - 1)**2)
            delta_err = new_err - old_err
            
            if delta_err < 0:
                T[i, :, k] = new_slice
                W_np[idx_w, r] += delta
                error += delta_err
                if error == 0:
                    return U_np.tolist(), V_np.tolist(), W_np.tolist(), 0
                        
    return U_np.tolist(), V_np.tolist(), W_np.tolist(), int(error)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to reduce rank from 35 down to 25
    # Start with Schoolbook (Rank 36)
    current_U, current_V, current_W = U_sb, V_sb, W_sb
    current_rank = 36
    
    for target_rank in range(35, 24, -1):
        if time.time() > deadline:
            break
        
        # Try to reduce current_rank to target_rank (which is current_rank - 1)
        # by zeroing out one column and repairing.
        
        success = False
        for col_idx in range(current_rank):
            if time.time() > deadline:
                break
                
            # Copy current solution
            U_try = [row[:] for row in current_U]
            V_try = [row[:] for row in current_V]
            W_try = [row[:] for row in current_W]
            
            # Zero out column col_idx
            for i in range(len(U_try)):
                U_try[i][col_idx] = 0
            for i in range(len(V_try)):
                V_try[i][col_idx] = 0
            for i in range(len(W_try)):
                W_try[i][col_idx] = 0
            
            # Run local search to repair
            U_res, V_res, W_res, err = _local_search(U_try, V_try, W_try, n, m, p, max_iters=1000, deadline=deadline)
            
            if err == 0:
                # Verify
                if _verify_exact(U_res, V_res, W_res, n, m, p):
                    # Remove the zeroed column to reduce rank
                    U_new = [[val for c, val in enumerate(row) if c != col_idx] for row in U_res]
                    V_new = [[val for c, val in enumerate(row) if c != col_idx] for row in V_res]
                    W_new = [[val for c, val in enumerate(row) if c != col_idx] for row in W_res]
                    
                    best_U, best_V, best_W = U_new, V_new, W_new
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank = target_rank
                    success = True
                    break # Found a reduction, try next lower rank
        
        if not success:
            # If we can't reduce from current_rank, we probably can't reduce further
            # But maybe a different column? We tried all columns.
            # So we break.
            break

    return {"U": best_U, "V": best_V, "W": best_W}
