import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T is sparse.
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    # Pre-compute target tensor and mask for fast error checking
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for u, v, w in target_indices:
        T_target[u, v, w] = 1
        
    def get_error(U, V, W):
        # Compute T_recon = sum_c U_c V_c W_c
        # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
        error = np.sum(np.abs(T_recon - T_target))
        return error

    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    best_solution = None
    best_rank = None
    
    # Try ranks from 18 upwards
    for r in range(18, 21):
        if time.time() - start_time > max_time:
            break
            
        # Run several random restarts for this rank
        num_restarts = 100 if r == 18 else 50
        
        for restart in range(num_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with sparse random integers
            U = np.zeros((dim_u, r), dtype=int)
            V = np.zeros((dim_v, r), dtype=int)
            W = np.zeros((dim_w, r), dtype=int)
            
            density = 0.4
            mask_u = np.random.rand(dim_u, r) < density
            mask_v = np.random.rand(dim_v, r) < density
            mask_w = np.random.rand(dim_w, r) < density
            
            U[mask_u] = np.random.randint(-1, 2, size=np.sum(mask_u))
            V[mask_v] = np.random.randint(-1, 2, size=np.sum(mask_v))
            W[mask_w] = np.random.randint(-1, 2, size=np.sum(mask_w))
            
            current_error = get_error(U, V, W)
            
            # Hill Climbing
            max_steps = 200
            for step in range(max_steps):
                if time.time() - start_time > max_time:
                    break
                    
                if current_error == 0:
                    best_rank = r
                    best_solution = (U, V, W)
                    break
                    
                # Pick a random matrix and entry to change
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    dim = dim_u
                elif mat_idx == 1:
                    mat = V
                    dim = dim_v
                else:
                    mat = W
                    dim = dim_w
                    
                row = np.random.randint(dim)
                col = np.random.randint(r)
                
                old_val = mat[row, col]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val:
                    continue
                    
                mat[row, col] = new_val
                new_error = get_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                else:
                    mat[row, col] = old_val
                    
            if best_solution is not None:
                break
                
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, column] = 1
                    V[v, column] = 1
                    W[w, column] = 1
                    column += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
