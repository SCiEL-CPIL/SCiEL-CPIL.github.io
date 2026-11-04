# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    def snap(x):
        # Snap to nearest half-integer
        return round(x * 2) / 2.0
    
    U_list = [[snap(x) for x in row] for row in U.tolist()]
    V_list = [[snap(x) for x in row] for row in V.tolist()]
    W_list = [[snap(x) for x in row] for row in W.tolist()]
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }

def _verify(U, V, W, n, m, p, tol=1e-8):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k, i in the tensor definition W[k*n+i, r]
    # So W_t[k, i, r]
    
    # Use einsum for speed
    # U_t[i, j, r]
    # V_t[j, k, r]
    # W_t[k, i, r] -> We need to align indices.
    # Let's transpose W_t to (i, k, r) to match i, j, k order?
    # W_t is (p, n, rank) which is (k, i, r).
    # We want T[i, j, k].
    # Term is U[i, j, r] * V[j, k, r] * W[k, i, r].
    # einsum: 'ijr,jkr,kir->ijk'
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False, 1.0
        
    error = np.max(np.abs(T - 1.0))
    return error < tol, error

def _objective(params, n, m, p, rank):
    # params is a flat array of size (n*m + m*p + p*n) * rank
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U_t = U_flat.reshape(n, m, rank)
    V_t = V_flat.reshape(m, p, rank)
    W_t = W_flat.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    error = T - 1.0
    return np.sum(error * error)

def _local_search_halfint(U, V, W, n, m, p, max_iters=200, deadline=None):
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Initial error
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = T - 1.0
    max_err = np.max(np.abs(err))
    
    if max_err < 1e-8:
        return U, V, W, max_err
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Find worst index
        worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
        wi, wj, wk = worst_idx
        werr = err[wi, wj, wk]
        
        # Try to adjust one entry in U, V, or W
        # We will iterate over all possible moves for the worst error term
        # Move in U[wi, wj, r]
        # Move in V[wj, wk, r]
        # Move in W[wk, wi, r]
        
        improved = False
        for r in range(rank):
            # Try U
            old_u = U_t[wi, wj, r]
            grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
            if grad_u != 0:
                # Try +/- 0.5
                for delta in [0.5, -0.5]:
                    new_val = old_u + delta
                    # Check if it's a valid half-integer (it should be if old is)
                    U_t[wi, wj, r] = new_val
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < max_err - 1e-10:
                        max_err = new_err
                        err = T - 1.0
                        improved = True
                        break
                    U_t[wi, wj, r] = old_u
            if improved: break
            
            if improved: break
            
            # Try V
            old_v = V_t[wj, wk, r]
            grad_v = U_t[wi, wj, r] * W_t[wk, wi, r]
            if grad_v != 0:
                for delta in [0.5, -0.5]:
                    new_val = old_v + delta
                    V_t[wj, wk, r] = new_val
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < max_err - 1e-10:
                        max_err = new_err
                        err = T - 1.0
                        improved = True
                        break
                    V_t[wj, wk, r] = old_v
            if improved: break
            
            if improved: break
            
            # Try W
            old_w = W_t[wk, wi, r]
            grad_w = U_t[wi, wj, r] * V_t[wj, wk, r]
            if grad_w != 0:
                for delta in [0.5, -0.5]:
                    new_val = old_w + delta
                    W_t[wk, wi, r] = new_val
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < max_err - 1e-10:
                        max_err = new_err
                        err = T - 1.0
                        improved = True
                        break
                    W_t[wk, wi, r] = old_w
            if improved: break
            
        if max_err < 1e-8:
            return U, V, W, max_err
            
        if not improved:
            # If no single step improved, try a random perturbation or break
            # To avoid getting stuck, we can break and rely on outer loop restarts
            # or try a larger random jump
            if np.random.random() < 0.05:
                # Random perturbation
                for _ in range(3):
                    mat = np.random.choice([U_t, V_t, W_t])
                    idx = np.random.randint(mat.shape[0])
                    r = np.random.randint(rank)
                    mat[idx, r] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err

    return U, V, W, max_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = _to_lists(U0, V0, W0)
    
    # Known lower bounds: Rank(3x3x3) = 23, Rank(3x3x4) is likely around 25-27.
    # We search from 25 upwards.
    # Note: The problem asks for exact half-integer decomposition.
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values, but scale them to be reasonable
            # Using small random values helps L-BFGS
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B to find a continuous solution
            # This is much more effective than random grid search for finding the "valley"
            try:
                res = minimize(_objective, params, args=(n, m, p, target_rank), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
            except Exception:
                continue
                
            params_opt = res.x
            size_u = n * m * target_rank
            size_v = m * p * target_rank
            size_w = p * n * target_rank
            
            U_opt = params_opt[:size_u].reshape(n * m, target_rank)
            V_opt = params_opt[size_u:size_u + size_v].reshape(m * p, target_rank)
            W_opt = params_opt[size_u + size_v:].reshape(p * n, target_rank)
            
            # Check continuous error
            U_t = U_opt.reshape(n, m, target_rank)
            V_t = V_opt.reshape(m, p, target_rank)
            W_t = W_opt.reshape(p, n, target_rank)
            T_cont = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            err_cont = np.max(np.abs(T_cont - 1.0))
            
            # If continuous error is small, try to snap to half-integers
            if err_cont < 0.5:
                # Snap to nearest half-integer
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                # Verify snapped solution
                valid, err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U_snap, V_snap, W_snap)
                    return best_result
                
                # If snapping failed, try local search on the snapped values
                # Or try local search on the continuous values? No, must be half-integers.
                # The local search works on half-integers.
                # Initialize local search with snapped values
                U_ls, V_ls, W_ls, err_ls = _local_search_halfint(U_snap, V_snap, W_snap, n, m, p, max_iters=100, deadline=deadline - 2)
                
                valid, err_final = _verify(U_ls, V_ls, W_ls, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U_ls, V_ls, W_ls)
                    return best_result

    return best_result
# EVOLVE-BLOCK-END
