import time
import numpy as np
from scipy.optimize import least_squares
import sys

DEADLINE = 120.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 10

def _construct_trivial(n, m, p):
    """
    Construct the trivial (schoolbook) tensor decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _tensor_check(U, V, W, n, m, p, tol=1e-9):
    """
    Check if the decomposition is valid.
    Condition: For all i, k: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j, j'}
    """
    I_m = np.eye(m)
    for i in range(n):
        for k in range(p):
            U_sub = U[i*m : i*m+m, :] # (m, R)
            V_indices = np.arange(m) * p + k
            V_sub = V[V_indices, :] # (m, R)
            W_vec = W[k*n + i, :] # (R,)
            
            # M[j, jp] = sum_r U_sub[j, r] * V_sub[jp, r] * W_vec[r]
            M = (U_sub * W_vec) @ V_sub.T
            
            if not np.allclose(M, I_m, atol=tol):
                return False
    return True

def _snap_to_half_integers(U, V, W, tol=1e-4):
    """
    Snap values to nearest half-integer if close.
    """
    def snap(M):
        # Multiply by 2, round, divide by 2
        return np.round(M * 2) / 2.0
    
    U_snap = snap(U)
    V_snap = snap(V)
    W_snap = snap(W)
    
    # Check if snapping introduced error
    if np.allclose(U, U_snap, atol=tol) and np.allclose(V, V_snap, atol=tol) and np.allclose(W, W_snap, atol=tol):
        return U_snap, V_snap, W_snap
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, time_budget=10.0):
    """
    Attempt to find a valid decomposition of the given target rank.
    Uses least-squares optimization on the flattened tensor factors.
    """
    if _check_deadline():
        return None
        
    start_time = time.time()
    
    # Initial guess: Trivial decomposition, but we need to handle the rank difference.
    # If target_rank < n*m*p, we can't just truncate.
    # Strategy: Start with a random initialization or a "compressed" trivial init.
    # A better init: Use the trivial decomposition but allow columns to be linear combinations.
    # Actually, let's start with a random sparse matrix or a scaled trivial one.
    
    # For small ranks, random init often fails. 
    # Let's try to construct a "block" init if possible, or just random.
    # Given the complexity, we'll use a random init with small variance, 
    # but we must ensure the scale is appropriate.
    
    # A more robust init for LSQ: 
    # If we assume the solution is sparse, random init is bad.
    # However, for 2x4x7, the rank is likely close to 50.
    # Let's try to optimize from a "near-trivial" state.
    
    # We can't easily map 56 dims to R dims with a trivial init if R < 56.
    # Instead, we initialize U, V, W with random values from a small distribution.
    # To help convergence, we can use the fact that the solution is likely sparse.
    # But LSQ doesn't handle sparsity well directly.
    
    # Alternative: Use the known 50-rank structure as a starting point if target_rank >= 50.
    # If target_rank < 50, we are in uncharted territory.
    
    # Let's generate a random init.
    np.random.seed(42) # Deterministic for reproducibility in debugging, but we want variation?
    # Actually, let's use a mix.
    
    U = np.random.randn(n*m, target_rank) * 0.5
    V = np.random.randn(m*p, target_rank) * 0.5
    W = np.random.randn(p*n, target_rank) * 0.5
    
    # Scale up? The trivial solution has 1s. 
    # If we use small values, the gradient might be small.
    # Let's try values around 1.
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    def residuals(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank :]
        
        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)
        
        res_list = []
        for i in range(n):
            for k in range(p):
                U_sub = U_mat[i*m : i*m+m, :]
                V_indices = np.arange(m) * p + k
                V_sub = V_mat[V_indices, :]
                W_vec = W_mat[k*n + i, :]
                
                M = (U_sub * W_vec) @ V_sub.T
                # Target is I_m
                res_list.append((M - np.eye(m)).flatten())
                
        return np.concatenate(res_list)
        
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use 'lm' method for least squares (requires full rank Jacobian, might fail if underdetermined)
        # 'trf' (Trust Region Reflective) is more robust for bounds/large problems.
        # We don't have bounds, so 'lm' is fast but fragile. 'trf' is safer.
        # Given the non-convexity, we might need multiple restarts.
        
        sol = least_squares(residuals, x0, method='trf', max_nfev=500, x_scale='jac')
        
        if time.time() - start_time > time_budget:
            return None
            
        if sol.success or sol.cost < 1e-6:
            U_opt = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = sol.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = sol.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Check validity
            if _tensor_check(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                # Try to snap to half-integers
                U_snap, V_snap, W_snap = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if _tensor_check(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    return U_snap, V_snap, W_snap
                else:
                    return U_opt, V_opt, W_opt
                    
    except Exception as e:
        pass
        
    return None

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    # Fallback: Trivial
    U_t, V_t, W_t = _construct_trivial(n, m, p)
    best_U, best_V, best_W = U_t, V_t, W_t
    best_rank = U_t.shape[1]
    
    # We know 50 is possible via blocks. 
    # Let's try to find something better than 50, or at least confirm 50 quickly.
    # Since the current program gets 50, we should aim for < 50.
    # Known bounds for 2x4x7: 
    # Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # Upper bound is 56 (trivial), 50 (blocks).
    # Is 49 possible? 48?
    
    # We will iterate downwards from 49.
    # To save time, we will limit the number of restarts per rank.
    
    max_restarts = 5
    for target_rank in range(49, 27, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Different random seeds for each restart
            np.random.seed(1000 + restart * 17 + target_rank)
            
            res = _try_rank_reduction(n, m, p, target_rank, time_budget=5.0)
            if res is not None:
                U_new, V_new, W_new = res
                # Final strict check
                if _tensor_check(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    found = True
                    break # Move to next lower rank
            
        if found:
            # If we found a solution for this rank, we could continue to lower ranks,
            # but it gets exponentially harder. 
            # Given the time budget, if we find one, it's a huge win.
            # Let's try to go one step lower if we have time.
            if time.time() - start_time > DEADLINE - 30:
                break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Constructed.")
