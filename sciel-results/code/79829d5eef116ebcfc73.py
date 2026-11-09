import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0

def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _build_tensor_indices(n, m, p):
    """Build the sparse tensor indices."""
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    return np.array(idx_u), np.array(idx_v), np.array(idx_w)

def _error_metric(params, rank, num_u, num_v, num_w, idx_u, idx_v, idx_w):
    """
    Calculate the squared error of the tensor decomposition.
    params is a flat array of size 3 * (num_u + num_v + num_w) * rank? 
    No, let's pass U, V, W separately or flatten them.
    Flattening: U (num_u x rank), V (num_v x rank), W (num_w x rank).
    Total size: (num_u + num_v + num_w) * rank.
    """
    # Unpack
    size_u = num_u * rank
    size_v = num_v * rank
    size_w = num_w * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u : size_u + size_v]
    W_flat = params[size_u + size_v : size_u + size_v + size_w]
    
    U = U_flat.reshape((num_u, rank))
    V = V_flat.reshape((num_v, rank))
    W = W_flat.reshape((num_w, rank))
    
    # Compute T_approx at non-zero indices
    # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # We only care about the specific indices (idx_u, idx_v, idx_w) where T_target = 1.
    # And we assume other entries should be 0, but for the error metric, 
    # minimizing the error at the non-zero positions is the primary goal for exactness 
    # if we later snap. However, to ensure the tensor is exactly the multiplication tensor,
    # we must ensure that T_approx is 1 at the non-zero spots and 0 elsewhere.
    # A full error metric would be expensive if we check all zeros.
    # But for small dimensions, we can check all.
    # T_full shape: num_u x num_v x num_w.
    # 9 x 24 x 24 = 5184 entries. This is small.
    
    T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Target tensor
    T_target = np.zeros((num_u, num_v, num_w))
    T_target[idx_u, idx_v, idx_w] = 1.0
    
    err = np.sum((T_full - T_target) ** 2)
    return err

def _try_rank(n, m, p, rank, time_limit, seed):
    """
    Try to find a decomposition of given rank using random restarts and Nelder-Mead.
    """
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    idx_u, idx_v, idx_w = _build_tensor_indices(n, m, p)
    
    np.random.seed(seed)
    
    best_params = None
    best_err = np.inf
    
    # Number of restarts depends on time
    # Each Nelder-Mead run might take ~0.1-0.5s.
    # Let's try 20 restarts.
    max_restarts = 20
    restart_time_budget = time_limit / max_restarts
    
    for r in range(max_restarts):
        if time.time() > time_limit:
            break
            
        # Initialize random parameters
        # Scale matters. Let's initialize with small random values.
        init_scale = 1.0
        x0 = np.random.randn((num_u + num_v + num_w) * rank) * init_scale
        
        # Define objective function
        def objective(params):
            return _error_metric(params, rank, num_u, num_v, num_w, idx_u, idx_v, idx_w)
        
        # Optimize
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                
                # If error is very small, we might have found it
                if best_err < 1e-8:
                    break
                    
        except Exception as e:
            continue
            
    if best_params is None or best_err > 1e-6:
        return None
        
    # Unpack best parameters
    size_u = num_u * rank
    size_v = num_v * rank
    size_w = num_w * rank
    
    U_flat = best_params[:size_u]
    V_flat = best_params[size_u : size_u + size_v]
    W_flat = best_params[size_u + size_v : size_u + size_v + size_w]
    
    U = U_flat.reshape((num_u, rank))
    V = V_flat.reshape((num_v, rank))
    W = W_flat.reshape((num_w, rank))
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify
    T_full = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
    T_target = np.zeros((num_u, num_v, num_w))
    T_target[idx_u, idx_v, idx_w] = 1.0
    
    if np.allclose(T_full, T_target, atol=1e-9):
        return U_snap, V_snap, W_snap
    else:
        return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Ranks to try
    # Schoolbook is 72.
    # Try lower ranks.
    ranks_to_try = [40, 44, 48, 52, 56, 60, 64, 68, 70]
    
    seed = 42
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE_SECONDS - SAFETY_MARGIN:
            break
            
        if rank >= best_rank:
            continue
            
        time_limit = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
        
        # Allocate a portion of the remaining time to this rank
        # If we have many ranks, we should spend less time on each.
        # Let's say we spend at most 10 seconds per rank.
        rank_time = min(10.0, time_limit)
        
        try:
            res = _try_rank(n, m, p, rank, rank_time, seed)
            if res is not None:
                U_snap, V_snap, W_snap = res
                best_result = {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                best_rank = rank
                # If we found a low rank, we might stop if it's very low?
                # Or we can try to go even lower?
                # The list is sorted ascending, so we are trying lower ranks first.
                # If we succeed at 40, we can stop because we want the minimum.
                break
        except Exception as e:
            pass
        
        seed += 1

    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
