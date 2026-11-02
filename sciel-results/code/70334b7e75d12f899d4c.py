import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication (Strassen rank 11)
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify the current solution
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T_global = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    c_row = a_row
                    c_col = b_col
                    T_global[a_row*m+a_col, b_row*p+b_col, c_col*n+c_row] += 1
    
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r])
    
    if not np.allclose(T_global, T_recon, atol=1e-6):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to find a lower-rank solution using random search with multiple restarts
    if time.time() < deadline - 40:
        try:
            # Build the global tensor
            T_global_flat = T_global.flatten()
            
            def compute_error(params, target_rank):
                U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                T_recon = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_recon += np.einsum('i,j,k->ijk', U_p[:,r], V_p[:,r], W_p[:,r])
                
                err = np.sum((T_global - T_recon)**2)
                return err
            
            # Try ranks 43 down to 40
            for target_rank in [43, 42, 41, 40]:
                if time.time() > deadline - 30:
                    break
                
                found = False
                for restart in range(5):
                    if time.time() > deadline - 30:
                        break
                    
                    np.random.seed(100 + restart)
                    U_opt = np.random.randn(n*m, target_rank) * 0.1
                    V_opt = np.random.randn(m*p, target_rank) * 0.1
                    W_opt = np.random.randn(p*n, target_rank) * 0.1
                    
                    x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                    
                    try:
                        result = minimize(compute_error, x0, args=(target_rank,), method='Nelder-Mead', 
                                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
                        
                        if result.fun < 1e-6:
                            # Found a valid solution!
                            U_new = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                            V_new = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                            W_new = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                            
                            # Round to half-integers
                            U_new = np.round(U_new * 2) / 2
                            V_new = np.round(V_new * 2) / 2
                            W_new = np.round(W_new * 2) / 2
                            
                            # Verify after rounding
                            T_recon_new = np.zeros((n*m, m*p, p*n))
                            for r in range(target_rank):
                                T_recon_new += np.einsum('i,j,k->ijk', U_new[:,r], V_new[:,r], W_new[:,r])
                            
                            if np.allclose(T_global, T_recon_new, atol=1e-6):
                                return {
                                    "U": U_new.tolist(),
                                    "V": V_new.tolist(),
                                    "W": W_new.tolist()
                                }
                            else:
                                # Use unrounded if it works
                                T_recon_new = np.zeros((n*m, m*p, p*n))
                                for r in range(target_rank):
                                    T_recon_new += np.einsum('i,j,k->ijk', U_new[:,r], V_new[:,r], W_new[:,r])
                                
                                if np.allclose(T_global, T_recon_new, atol=1e-6):
                                    return {
                                        "U": U_new.tolist(),
                                        "V": V_new.tolist(),
                                        "W": W_new.tolist()
                                    }
                    except Exception:
                        continue
        
        except Exception:
            pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
