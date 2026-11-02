import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def get_residuals(U, V, W):
        res = np.zeros(len(triplets))
        for idx, (i, j, k) in enumerate(triplets):
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[idx] = val - 1.0
        return res
    
    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return np.sum(res * res)
    
    def optimize_with_time_limit(U, V, W, time_limit=2.0, max_iters=200):
        # Flatten parameters
        nU = U.size
        nV = V.size
        nW = W.size
        total_size = nU + nV + nW
        
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(params):
            U_flat = params[:nU].reshape(U.shape)
            V_flat = params[nU:nU+nV].reshape(V.shape)
            W_flat = params[nU+nV:nU+nV+nW].reshape(W.shape)
            return U_flat, V_flat, W_flat
        
        def objective(params):
            U_flat, V_flat, W_flat = unflatten_params(params)
            return total_error(U_flat, V_flat, W_flat)
        
        params0 = flatten_params(U, V, W)
        
        # Use L-BFGS-B with a time limit
        start_time = time.time()
        
        # Simple gradient descent with time limit
        lr = 0.01
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            res = get_residuals(U_curr, V_curr, W_curr)
            err = np.sum(res * res)
            if err < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U_curr)
            dV = np.zeros_like(V_curr)
            dW = np.zeros_like(W_curr)
            
            for idx, (i, j, k) in enumerate(triplets):
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                r = res[idx]
                term_VW = V_curr[vi] * W_curr[wi]
                term_UW = U_curr[ui] * W_curr[wi]
                term_UV = U_curr[ui] * V_curr[vi]
                
                dU[ui] += 2 * r * term_VW
                dV[vi] += 2 * r * term_UW
                dW[wi] += 2 * r * term_UV
            
            # Update with adaptive learning rate
            U_curr -= lr * dU
            V_curr -= lr * dV
            W_curr -= lr * dW
        
        return U_curr, V_curr, W_curr
    
    def schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        return U, V, W
    
    def randomized_algebraic_search(max_restarts_per_rank=30, time_per_opt=1.5):
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook()
        best_err = 0.0
        
        start_time = time.time()
        deadline = 110  # 10s margin
        
        # Try ranks from 40 to 47
        for r in range(40, 48):
            if time.time() > deadline:
                break
            
            for restart in range(max_restarts_per_rank):
                if time.time() > deadline:
                    break
                
                # Initialize with schoolbook terms (first r) plus noise
                U_init, V_init, W_init = schoolbook(r)
                
                # Add noise to help escape local minima
                noise_level = 0.5
                U_init = U_init + noise_level * np.random.randn(n*m, r)
                V_init = V_init + noise_level * np.random.randn(m*p, r)
                W_init = W_init + noise_level * np.random.randn(p*n, r)
                
                # Run optimization with time limit
                U_opt, V_opt, W_opt = optimize_with_time_limit(U_init, V_init, W_init, time_limit=time_per_opt)
                
                err = total_error(U_opt, V_opt, W_opt)
                
                if err < 1e-8:
                    # Success!
                    return U_opt, V_opt, W_opt, r
                
                # Keep the best found so far
                if best_err == 0.0 or err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
        
        return best_U, best_V, best_W, best_rank
    
    U, V, W, rank = randomized_algebraic_search()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
