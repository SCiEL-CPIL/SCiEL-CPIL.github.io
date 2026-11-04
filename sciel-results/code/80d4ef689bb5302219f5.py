"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin before 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions using optimization
    # The tensor is T[i,j,k] = 1 if we're multiplying (i,j) element of A by (j,k) element of B
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to equal the tensor
    
    best_result = None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual tensor rank for 3x4*4x6 is likely higher
    # Let's try from 50 up to see if we can find anything below 72
    
    for target_rank in range(50, min(rank_schoolbook, 72)):
        if time.time() > deadline - 30:  # Stop if less than 30s remaining
            break
            
        # Try multiple random initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
                
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Optimize to minimize ||tensor - decomposition||
            from scipy.optimize import minimize
            
            def objective(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute decomposition
                # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a tensor contraction
                result = np.zeros((n, m, p))
                for r in range(target_rank):
                    u_col = U_flat[:, r]
                    v_col = V_flat[:, r]
                    w_col = W_flat[:, r]
                    # u_col has shape (n*m,), v_col has shape (m*p,), w_col has shape (p*n,)
                    # Reshape and multiply
                    u_mat = u_col.reshape(n, m)
                    v_mat = v_col.reshape(m, p)
                    w_mat = w_col.reshape(p, n)
                    # For each (i,j,k): u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                result[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                
                # The target tensor is all 1s
                error = np.sum((result - 1.0)**2)
                return error
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result_opt = minimize(objective, x0, method='L-BFGS-B', 
                                     options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result_opt.fun < 1e-6:  # Found a valid decomposition
                    params = result_opt.x
                    U_opt = params[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Snap to rational values if possible
                    def snap_to_rational(arr):
                        arr_snapped = arr.copy()
                        for idx in np.ndindex(arr.shape):
                            val = arr[idx]
                            # Try to snap to simple fractions
                            for denom in [1, 2, 3, 4, 6]:
                                for num in range(-10, 11):
                                    if abs(val - num/denom) < 1e-4:
                                        arr_snapped[idx] = num/denom
                                        break
                                else:
                                    continue
                                break
                        return arr_snapped
                    
                    U_snapped = snap_to_rational(U_opt)
                    V_snapped = snap_to_rational(V_opt)
                    W_snapped = snap_to_rational(W_opt)
                    
                    # Verify the snapped version
                    result_check = np.zeros((n, m, p))
                    for r in range(target_rank):
                        u_mat = U_snapped[:, r].reshape(n, m)
                        v_mat = V_snapped[:, r].reshape(m, p)
                        w_mat = W_snapped[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    result_check[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                    
                    if np.allclose(result_check, 1.0, atol=1e-4):
                        best_result = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        found = True
                        break
                        
            except Exception as e:
                pass
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Return schoolbook decomposition
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
