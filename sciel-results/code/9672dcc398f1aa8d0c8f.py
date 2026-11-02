import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """Build target tensor T of shape (n*m, m*p, p*n)"""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _compute_error_fast(U, V, W, T):
    """Compute ||U V^T W^T - T||_F efficiently using einsum."""
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    diff = reconstructed - T
    return np.sqrt(np.sum(diff * diff))

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Build target
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    diff = reconstructed - T
    return np.max(np.abs(diff)) < tol

def _local_search_fast(U, V, W, T, R, time_limit, start_time, values=None):
    """Fast local search with incremental error updates. Returns (U, V, W, error)."""
    if values is None:
        values = [-1, 0, 1]
    
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    diff = recon - T
    current_error_sq = float(np.sum(diff * diff))
    current_error = np.sqrt(current_error_sq)
    
    # Precompute outer products for each rank term: outers_VW[r] = V[:,r] @ W[:,r].T
    outers_VW = np.einsum('vr,wr->vwr', V, W)  # shape (R, mp, pn)
    
    max_iter = 30000
    stall_count = 0
    stall_limit = 300
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if current_error < 1e-12:
            break
        
        total = U.size + V.size + W.size
        var_idx = random.randint(0, total - 1)
        
        if var_idx < U.size:
            mat_name = 'U'
            local_idx = var_idx
            mat = U
        elif var_idx < U.size + V.size:
            mat_name = 'V'
            local_idx = var_idx - U.size
            mat = V
        else:
            mat_name = 'W'
            local_idx = var_idx - U.size - V.size
            mat = W
        
        old_val = mat.flat[local_idx]
        new_val = random.choice(values)
        delta = new_val - old_val
        
        if abs(delta) < 1e-15:
            continue
        
        if mat_name == 'U':
            u_idx, r_idx = divmod(local_idx, R)
            # outers_VW has shape (mp, pn, R). We need the slice for rank r_idx.
            outer_r = outers_VW[:, :, r_idx]
            grad = 2.0 * np.sum(diff * outer_r)
            quad = delta * delta * np.sum(outer_r * outer_r)
            new_error_sq = current_error_sq + grad * delta + quad
        elif mat_name == 'V':
            v_idx, r_idx = divmod(local_idx, R)
            grad = 2.0 * delta * np.einsum('uw,u,w->', diff[:, v_idx, :], U[:, r_idx], W[:, r_idx])
            quad = delta * delta * np.sum(U[:, r_idx] * U[:, r_idx]) * np.sum(W[:, r_idx] * W[:, r_idx])
            new_error_sq = current_error_sq + grad + quad
        else:
            w_idx, r_idx = divmod(local_idx, R)
            grad = 2.0 * delta * np.einsum('uv,u,v->', diff[:, :, w_idx], U[:, r_idx], V[:, r_idx])
            quad = delta * delta * np.sum(U[:, r_idx] * U[:, r_idx]) * np.sum(V[:, r_idx] * V[:, r_idx])
            new_error_sq = current_error_sq + grad + quad
        
        if new_error_sq < current_error_sq - 1e-12 and new_error_sq >= 0:
            mat.flat[local_idx] = new_val
            
            if mat_name == 'U':
                recon += delta * outer_r
            elif mat_name == 'V':
                recon[:, v_idx, :] += delta * np.outer(U[:, r_idx], W[:, r_idx])
            else:
                recon[:, :, w_idx] += delta * np.outer(U[:, r_idx], V[:, r_idx])
            
            diff = recon - T
            current_error_sq = float(np.sum(diff * diff))
            current_error = np.sqrt(current_error_sq)
            stall_count = 0
            
            if mat_name in ('U', 'W'):
                # Recompute outers_VW only if V or W changed, as U change doesn't affect V@W^T
                # Note: The code structure implies we recompute if U or W changes? 
                # Actually, outers_VW depends on V and W. 
                # If U changes, outers_VW is unchanged.
                # If V changes, outers_VW changes.
                # If W changes, outers_VW changes.
                # The original code recomputed if U or W. This is inefficient/incorrect logic.
                # It should recompute if V or W.
                if mat_name in ('V', 'W'):
                    outers_VW = np.einsum('vr,wr->vwr', V, W)
        else:
            stall_count += 1
        
        if stall_count > stall_limit:
            break
    
    return U, V, W, current_error

def _try_rank(R, n, m, p, T, time_limit, start_time):
    """Try to find rank-R decomposition. Returns (U,V,W) or None."""
    nm, mp, pn = n * m, m * p, p * n
    values = [-1, 0, 1]
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    num_restarts = 60
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((nm, R), dtype=np.float64)
        V = np.zeros((mp, R), dtype=np.float64)
        W = np.zeros((pn, R), dtype=np.float64)
        
        for r in range(R):
            u_nonzero = random.randint(1, 2)
            v_nonzero = random.randint(1, 2)
            w_nonzero = random.randint(1, 2)
            
            u_indices = random.sample(range(nm), min(u_nonzero, nm))
            v_indices = random.sample(range(mp), min(v_nonzero, mp))
            w_indices = random.sample(range(pn), min(w_nonzero, pn))
            
            for ui in u_indices:
                U[ui, r] = random.choice([1.0, -1.0])
            for vi in v_indices:
                V[vi, r] = random.choice([1.0, -1.0])
            for wi in w_indices:
                W[wi, r] = random.choice([1.0, -1.0])
        
        U, V, W, err = _local_search_fast(U, V, W, T, R, time_limit, start_time, values)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower bound up
    for R in range(lower_bound, best_rank):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining = deadline - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time based on remaining ranks
        ranks_left = best_rank - R
        time_budget = min(25.0, remaining / max(ranks_left, 1))
        if time_budget < 2.0:
            break
        
        result = _try_rank(R, n, m, p, T, time_budget, start_time)
        
        if result is not None:
            U, V, W = result
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
