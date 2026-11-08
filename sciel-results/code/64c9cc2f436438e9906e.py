import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Value set for half-integer entries
    values = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    # Search for lower rank
    # Start from a reasonable lower bound. 
    # R(2,4,7) is likely < 40.
    for r in range(15, rank_sb):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with half-integers
            U = np.random.choice(values, size=(n * m, r))
            V = np.random.choice(values, size=(m * p, r))
            W = np.random.choice(values, size=(p * n, r))
            
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            # Precompute T
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            error = np.sum((T - 1) ** 2)
            
            if error < 1e-9:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
            
            # Local Search
            improved = True
            stall_count = 0
            max_stall = 100
            max_iters = 2000
            iter_count = 0
            
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                if time.time() > deadline:
                    break
                    
                var_type = np.random.randint(3)
                
                if var_type == 0: # Change U[i,j,c]
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    c = np.random.randint(r)
                    old_val = U_3d[i, j, c]
                    
                    # Try all values in 'values'
                    best_new_val = None
                    best_new_err = error
                    best_T_slice = None
                    best_delta_T = None
                    
                    for new_val in values:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        
                        delta_T = diff * (V_3d[j, :, c] * W_3d[:, i, c])
                        T_new = T[i, j, :] + delta_T
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        old_err_slice = np.sum((T[i, j, :] - 1) ** 2)
                        
                        new_err_total = error + new_err_slice - old_err_slice
                        
                        if new_err_total < best_new_err - 1e-12:
                            best_new_err = new_err_total
                            best_new_val = new_val
                            best_T_slice = T_new
                            best_delta_T = delta_T
                            
                    if best_new_val is not None:
                        T[i, j, :] = best_T_slice
                        U_3d[i, j, c] = best_new_val
                        error = best_new_err
                        improved = True
                            
                elif var_type == 1: # Change V[j,k,c]
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    c = np.random.randint(r)
                    old_val = V_3d[j, k, c]
                    
                    best_new_val = None
                    best_new_err = error
                    best_T_slice = None
                    best_delta_T = None
                    
                    for new_val in values:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        
                        delta_T = (U_3d[:, j, c] * W_3d[k, :, c]) * diff
                        T_new = T[:, j, k] + delta_T
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        old_err_slice = np.sum((T[:, j, k] - 1) ** 2)
                        
                        new_err_total = error + new_err_slice - old_err_slice
                        
                        if new_err_total < best_new_err - 1e-12:
                            best_new_err = new_err_total
                            best_new_val = new_val
                            best_T_slice = T_new
                            best_delta_T = delta_T
                            
                    if best_new_val is not None:
                        T[:, j, k] = best_T_slice
                        V_3d[j, k, c] = best_new_val
                        error = best_new_err
                        improved = True
                            
                else: # Change W[k,i,c]
                    k = np.random.randint(p)
                    i = np.random.randint(n)
                    c = np.random.randint(r)
                    old_val = W_3d[k, i, c]
                    
                    best_new_val = None
                    best_new_err = error
                    best_T_slice = None
                    best_delta_T = None
                    
                    for new_val in values:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        
                        delta_T = (U_3d[i, :, c] * V_3d[:, k, c]) * diff
                        T_new = T[i, :, k] + delta_T
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        old_err_slice = np.sum((T[i, :, k] - 1) ** 2)
                        
                        new_err_total = error + new_err_slice - old_err_slice
                        
                        if new_err_total < best_new_err - 1e-12:
                            best_new_err = new_err_total
                            best_new_val = new_val
                            best_T_slice = T_new
                            best_delta_T = delta_T
                            
                    if best_new_val is not None:
                        T[i, :, k] = best_T_slice
                        W_3d[k, i, c] = best_new_val
                        error = best_new_err
                        improved = True
                            
                if not improved:
                    stall_count += 1
                    if stall_count > max_stall:
                        # Break out of local search, try new restart
                        break
                else:
                    stall_count = 0
                    
            if error < 1e-9:
                best_rank = r
                best_U = U_3d.flatten().reshape(n*m, r)
                best_V = V_3d.flatten().reshape(m*p, r)
                best_W = W_3d.flatten().reshape(p*n, r)
                found = True
                break
        
        if found:
            break

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
