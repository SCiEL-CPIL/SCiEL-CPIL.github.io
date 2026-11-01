import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Precompute indices for faster error calculation if needed, 
    # but einsum is fast enough for small dims.
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
        
    def check_exact(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=1e-9)
        
    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2
        
    def discrete_local_search(U, V, W, max_iters=500):
        """
        Greedy local search on the half-integer grid.
        Tries to flip one entry in U, V, or W to a nearby half-integer 
        value that reduces the error.
        """
        current_err = compute_error(U, V, W)
        if current_err < 1e-9:
            return U, V, W, True
            
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Candidates for new values: current +/- 0.5, current +/- 1.0
        # Since we start from snapped values, they are multiples of 0.5.
        # Neighbors are also multiples of 0.5.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            # Try flipping entries in U
            for i in range(U.shape[0]):
                for k in range(U.shape[1]):
                    old_val = U[i, k]
                    # Try neighbors: old_val +/- 0.5
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        # Optional: bound the values to avoid explosion
                        if abs(new_val) > 5: continue
                        U[i, k] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            U[i, k] = old_val
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved: continue
            
            # Try flipping entries in V
            for i in range(V.shape[0]):
                for k in range(V.shape[1]):
                    old_val = V[i, k]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5: continue
                        V[i, k] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            V[i, k] = old_val
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved: continue
            
            # Try flipping entries in W
            for i in range(W.shape[0]):
                for k in range(W.shape[1]):
                    old_val = W[i, k]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5: continue
                        W[i, k] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            W[i, k] = old_val
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                break
                
        return U, V, W, False

    def try_rank(r, num_restarts=20):
        if time.time() > deadline:
            return None
        
        size = r * (n*m + m*p + p*n)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            # Using a mix of small and slightly larger values can help escape local minima
            scale = 0.5 + 0.5 * (restart % 5) / 5.0
            params = np.random.randn(size) * scale
            
            def objective(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                return np.sum((T_approx - target)**2)
            
            def gradient(params):
                U = params[:r*n*m].reshape(n*m, r)
                V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = params[r*(n*m+m*p):].reshape(p*n, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000})
            except Exception:
                continue
                
            if time.time() > deadline:
                break
                
            if res.fun < 1e-6:
                U = res.x[:r*n*m].reshape(n*m, r)
                V = res.x[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                W = res.x[r*(n*m+m*p):].reshape(p*n, r)
                
                # Snap to half-integers
                U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
                
                if check_exact(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
                
                # If not exact, try discrete local search
                U_ls, V_ls, W_ls, found = discrete_local_search(U_snap, V_snap, W_snap, max_iters=200)
                if found:
                    return U_ls, V_ls, W_ls
                elif check_exact(U_ls, V_ls, W_ls):
                    return U_ls, V_ls, W_ls
        
        return None

    # Try ranks in increasing order
    for r in [20, 21, 22, 23]:
        res = try_rank(r, num_restarts=15)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 24
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
