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
                
    def compute_error_fast(U, V, W):
        """Compute squared error using efficient einsum"""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = T_approx - target
        return np.sum(diff * diff)
        
    def check_exact(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=1e-9)
        
    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2
        
    def discrete_local_search_fast(U, V, W, max_iters=300):
        """
        Faster greedy local search on the half-integer grid.
        Uses incremental error updates where possible.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = compute_error_fast(U, V, W)
        if current_err < 1e-9:
            return U, V, W, True
        
        # Precompute some intermediate values for faster updates
        # For U[i,k] change by delta: error change = 2*delta*sum_jc(V[j,k]*W[c,k]*target[i,j,c]) - delta^2*sum_jc(V[j,k]^2*W[c,k]^2)
        # This is complex, so we'll just do direct computation but with early exit
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try flipping entries in U
            for i in range(U.shape[0]):
                if improved:
                    break
                for k in range(U.shape[1]):
                    if improved:
                        break
                    old_val = U[i, k]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 4:
                            continue
                        U[i, k] = new_val
                        new_err = compute_error_fast(U, V, W)
                        if new_err < current_err - 1e-10:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            U[i, k] = old_val
                            if improved:
                                break
            
            if improved:
                continue
            
            # Try flipping entries in V
            for i in range(V.shape[0]):
                if improved:
                    break
                for k in range(V.shape[1]):
                    if improved:
                        break
                    old_val = V[i, k]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 4:
                            continue
                        V[i, k] = new_val
                        new_err = compute_error_fast(U, V, W)
                        if new_err < current_err - 1e-10:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            V[i, k] = old_val
                            if improved:
                                break
            
            if improved:
                continue
            
            # Try flipping entries in W
            for i in range(W.shape[0]):
                if improved:
                    break
                for k in range(W.shape[1]):
                    if improved:
                        break
                    old_val = W[i, k]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 4:
                            continue
                        W[i, k] = new_val
                        new_err = compute_error_fast(U, V, W)
                        if new_err < current_err - 1e-10:
                            current_err = new_err
                            improved = True
                            if current_err < 1e-9:
                                return U, V, W, True
                        else:
                            W[i, k] = old_val
                            if improved:
                                break
                    
            if not improved:
                break
                
        return U, V, W, False

    def try_rank(r, num_restarts=25):
        if time.time() > deadline:
            return None
        
        size = r * (n*m + m*p + p*n)
        nm, mp, pn = n*m, m*p, p*n
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Use structured initialization that might be closer to integer solutions
            # Mix of strategies: random, structured, and scaled
            strategy = restart % 3
            if strategy == 0:
                # Random small values
                scale = 0.3 + 0.4 * (restart % 5) / 5.0
                params = np.random.randn(size) * scale
            elif strategy == 1:
                # Structured: try to build from schoolbook-like patterns
                params = np.zeros(size)
                U_init = params[:r*nm].reshape(nm, r)
                V_init = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W_init = params[r*(nm+mp):].reshape(pn, r)
                
                # Initialize with sparse random patterns
                for col in range(r):
                    if col < 12:
                        # First 12 columns: schoolbook-like
                        idx = col
                        i = idx // 12
                        j = (idx // 4) % 3
                        k = idx % 4
                        if i < n and j < m and k < p:
                            U_init[i*m+j, col] = 1.0
                            V_init[j*p+k, col] = 1.0
                            W_init[k*n+i, col] = 1.0
                    else:
                        # Remaining columns: random small values
                        U_init[:, col] = np.random.randn(nm) * 0.3
                        V_init[:, col] = np.random.randn(mp) * 0.3
                        W_init[:, col] = np.random.randn(pn) * 0.3
            else:
                # Larger random initialization
                scale = 0.8 + 0.4 * (restart % 3) / 3.0
                params = np.random.randn(size) * scale
            
            def objective(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                return np.sum(diff * diff)
            
            def gradient(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            # Use L-BFGS-B with tighter tolerances
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800})
            except Exception:
                continue
                
            if time.time() > deadline:
                break
                
            # Check if we got close enough
            if res.fun < 1e-8:
                U = res.x[:r*nm].reshape(nm, r)
                V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                W = res.x[r*(nm+mp):].reshape(pn, r)
                
                # Try multiple snapping strategies
                # Strategy 1: Direct rounding
                U_snap1, V_snap1, W_snap1 = snap_to_half_integers(U, V, W)
                if check_exact(U_snap1, V_snap1, W_snap1):
                    return U_snap1, V_snap1, W_snap1
                
                # Strategy 2: Try small perturbations before snapping
                found = False
                for pert_scale in [0.1, 0.2, 0.3, 0.5]:
                    U_pert = U + np.random.randn(*U.shape) * pert_scale
                    V_pert = V + np.random.randn(*V.shape) * pert_scale
                    W_pert = W + np.random.randn(*W.shape) * pert_scale
                    U_snap2, V_snap2, W_snap2 = snap_to_half_integers(U_pert, V_pert, W_pert)
                    if check_exact(U_snap2, V_snap2, W_snap2):
                        return U_snap2, V_snap2, W_snap2
                    found = True
                
                # If not exact, try discrete local search from the snapped version
                U_ls, V_ls, W_ls, success = discrete_local_search_fast(U_snap1, V_snap1, W_snap1, max_iters=150)
                if success:
                    return U_ls, V_ls, W_ls
                elif check_exact(U_ls, V_ls, W_ls):
                    return U_ls, V_ls, W_ls
        
        return None

    # Try ranks in increasing order with adaptive restart counts
    # Spend more time on lower ranks if we're finding solutions
    rank_configs = [
        (20, 20),
        (21, 25),
        (22, 30),
        (23, 35),
    ]
    
    for r, num_restarts in rank_configs:
        if time.time() > deadline - 10:
            break
        res = try_rank(r, num_restarts=num_restarts)
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
