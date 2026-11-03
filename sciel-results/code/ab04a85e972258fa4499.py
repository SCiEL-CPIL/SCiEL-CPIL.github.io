import time
import sys
import numpy as np
import scipy.optimize as opt
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def try_rank_reduction(U0, V0, W0, target_rank, max_iters=5000):
    """Try to reduce rank by finding a lower-rank solution via optimization."""
    n, m, p = 2, 4, 4
    orig_rank = len(U0[0])
    
    # Convert to numpy for efficient operations
    U = np.array(U0, dtype=float)
    V = np.array(V0, dtype=float)
    W = np.array(W0, dtype=float)
    
    # Build the target tensor
    T_target = np.ones((n, m, p), dtype=float)
    
    # Try rank from target_rank up to orig_rank
    for rank in range(target_rank, orig_rank + 1):
        if time_up():
            break
            
        best_result = None
        best_err = float('inf')
        
        for init_seed in range(5):
            if time_up():
                break
                
            np.random.seed(init_seed)
            
            # Initialize with structured approach: use subset of original + random
            U_init = np.zeros((n*m, rank))
            V_init = np.zeros((m*p, rank))
            W_init = np.zeros((p*n, rank))
            
            # Copy first min(rank, orig_rank) columns from original
            copy_rank = min(rank, orig_rank)
            U_init[:, :copy_rank] = U[:, :copy_rank]
            V_init[:, :copy_rank] = V[:, :copy_rank]
            W_init[:, :copy_rank] = W[:, :copy_rank]
            
            # Add small random perturbation for seeds > 0
            if init_seed > 0:
                scale = 0.05 * init_seed
                U_init += scale * np.random.randn(n*m, rank)
                V_init += scale * np.random.randn(m*p, rank)
                W_init += scale * np.random.randn(p*n, rank)
            
            # Define the objective function
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute T_recon efficiently
                T_recon = np.zeros((n, m, p), dtype=float)
                for r in range(rank):
                    u_col = U_flat[:, r]
                    v_col = V_flat[:, r]
                    w_col = W_flat[:, r]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                
                return np.sum((T_recon - T_target)**2)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.minimize(objective, x0, method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if result.fun < best_err:
                    best_err = result.fun
                    best_result = result.x.copy()
                    
                    if best_err < 1e-10:
                        U_final = best_result[:n*m*rank].reshape(n*m, rank)
                        V_final = best_result[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_final = best_result[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        # Try rounding to half-integers
                        U_rounded = np.round(U_final * 2) / 2
                        V_rounded = np.round(V_final * 2) / 2
                        W_rounded = np.round(W_final * 2) / 2
                        
                        valid, err = verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        if valid:
                            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                        
                        return U_final.tolist(), V_final.tolist(), W_final.tolist()
                        
            except Exception:
                continue
        
        if best_result is not None and best_err < 1e-8:
            U_final = best_result[:n*m*rank].reshape(n*m, rank)
            V_final = best_result[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_final = best_result[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return U_final.tolist(), V_final.tolist(), W_final.tolist()
    
    return None

def construct():
    n, m, p = 2, 4, 4
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        # Try to reduce the rank from 28 down to 24
        for target_rank in [24, 25, 26, 27]:
            if time_up():
                break
                
            result = try_rank_reduction(U, V, W, target_rank)
            if result is not None:
                U_new, V_new, W_new = result
                valid_new, err_new = verify_decomposition(U_new, V_new, W_new)
                if valid_new:
                    return {"U": U_new, "V": V_new, "W": W_new}
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback naive
    rank = 32
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][idx] = 1.0
                V[j*p+k][idx] = 1.0
                W[k*n+i][idx] = 1.0
                idx += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
