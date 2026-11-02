import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    
    error = np.linalg.norm(reconstructed - T)
    return error

def _compute_error_squared(U, V, W, T):
    """Compute squared Frobenius norm error (faster, no sqrt)."""
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    diff = reconstructed - T
    return np.sum(diff * diff)

def _gradient_guided_search(U, V, W, T, values, max_iter=50000, timeout=10):
    """
    Gradient-guided coordinate descent with proper state management.
    Uses gradient information to prioritize which coordinates to try.
    """
    start_time = time.time()
    current_err_sq = _compute_error_squared(U, V, W, T)
    current_error = np.sqrt(current_err_sq) if current_err_sq < float('inf') else float('inf')
    
    n_u, n_v, n_w = U.size, V.size, W.size
    no_improve_count = 0
    max_no_improve = 500
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0.0:
            break
        
        # Compute gradients for all variables to guide search
        # d(err^2)/d(U[r,c]) = 2 * sum_{v,w} T'_{v,w} * V[v,c] * W[w,c]
        # where T' = reconstructed - T
        
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = reconstructed - T  # shape (n*m, m*p, p*n)
        
        # Gradient for U: (n*m, R)
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        # Gradient for V: (m*p, R)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
        # Gradient for W: (p*n, R)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
        
        # Combine gradients into a single array for ranking
        all_grads = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        all_mats = [(U, 0, n_u), (V, n_u, n_u + n_v), (W, n_u + n_v, n_u + n_v + n_w)]
        
        # Select top K coordinates by |gradient| to try
        K = 20
        top_indices = np.argsort(np.abs(all_grads))[-K:]
        
        # Shuffle among top K for diversity
        random.shuffle(top_indices.tolist())
        
        best_new_err_sq = current_err_sq
        best_move = None
        best_new_error = current_error
        
        for idx in top_indices:
            # Determine which matrix and position
            if idx < n_u:
                mat = U
                mat_idx = idx
                grad_val = grad_U.ravel()[idx]
            elif idx < n_u + n_v:
                mat = V
                mat_idx = idx - n_u
                grad_val = grad_V.ravel()[idx - n_u]
            else:
                mat = W
                mat_idx = idx - n_u - n_v
                grad_val = grad_W.ravel()[idx - n_u - n_v]
            
            old_val = mat.flat[mat_idx]
            
            # Try a few candidate new values around the gradient direction
            # The gradient tells us which direction to move
            candidates = []
            if grad_val > 0:
                # Move in negative direction to reduce error
                candidates = [old_val - 0.5, old_val - 1.0, old_val - 1.5, old_val - 2.0]
            elif grad_val < 0:
                # Move in positive direction to reduce error
                candidates = [old_val + 0.5, old_val + 1.0, old_val + 1.5, old_val + 2.0]
            else:
                candidates = [old_val - 1.0, old_val + 1.0, old_val - 0.5, old_val + 0.5]
            
            # Also try some random values
            candidates.append(random.choice(values))
            candidates.append(random.choice(values))
            
            # Deduplicate and filter
            seen = set()
            unique_candidates = []
            for c in candidates:
                # Round to nearest half-integer for exact representation
                c_rounded = round(c * 2) / 2
                if c_rounded != old_val and c_rounded not in seen:
                    seen.add(c_rounded)
                    unique_candidates.append(c_rounded)
            
            for new_val in unique_candidates[:3]:  # Limit to 3 candidates per variable
                mat.flat[mat_idx] = new_val
                new_err_sq = _compute_error_squared(U, V, W, T)
                
                if new_err_sq < best_new_err_sq:
                    best_new_err_sq = new_err_sq
                    best_new_error = np.sqrt(best_new_err_sq)
                    best_move = (mat, mat_idx, old_val, new_val)
                
                # Revert
                mat.flat[mat_idx] = old_val
        
        if best_move is not None:
            mat, mat_idx, old_val, new_val = best_move
            mat.flat[mat_idx] = new_val
            current_err_sq = best_new_err_sq
            current_error = best_new_error
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return current_error

def _structured_init(n, m, p, R, seed):
    """
    Create a structured initialization based on Kronecker product patterns.
    For 2x3x4 matrix multiplication, try to exploit the structure.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strategy: Use a mix of structured and random initialization
    # The tensor has 24 non-zero entries. Try to cover them with R terms.
    
    # Generate random half-integer values
    def rand_half_int(size):
        return (rng.randint(0, 2 * R + 1, size=size) - R) / 2.0
    
    # Fill with small random values
    for c in range(R):
        # Each column (rank term) should have some structure
        # U column: (n*m, ) = (6,)
        # V column: (m*p, ) = (12,)
        # W column: (p*n, ) = (8,)
        
        # Random sparse initialization
        u_vals = rand_half_int(n * m)
        v_vals = rand_half_int(m * p)
        w_vals = rand_half_int(p * n)
        
        # Make them sparse (many zeros)
        u_mask = rng.rand(n * m) < 0.4
        v_mask = rng.rand(m * p) < 0.3
        w_mask = rng.rand(p * n) < 0.35
        
        u_vals[~u_mask] = 0
        v_vals[~v_mask] = 0
        w_vals[~w_mask] = 0
        
        U[:, c] = u_vals
        V[:, c] = v_vals
        W[:, c] = w_vals
    
    return U, V, W

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using gradient-guided search.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 8
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        # Structured initialization
        U, V, W = _structured_init(n, m, p, R, seed=i * 12345 + 678)
        
        err = _compute_decomposition_error(U, V, W, T)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error == 0.0:
            break
        
        # Gradient-guided local search
        search_time = min(remaining_time - 0.2, time_per_restart)
        if search_time > 0.3:
            err = _gradient_guided_search(U, V, W, T, values, max_iter=30000, timeout=search_time)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error == 0.0:
            break
    
    if best_error < 1e-10:
        # Verify exactness
        err = _compute_decomposition_error(best_U, best_V, best_W, T)
        if err < 1e-10:
            return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Try ranks lower than schoolbook, starting from lower values
    # Theoretical lower bound for 2x3x4 is at least 14 (from Strassen-type bounds)
    # Try from 14 upward
    for R in range(14, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
        
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        # Allocate time: more time for lower ranks (more likely to be optimal)
        ranks_to_try = rank_schoolbook - R
        time_budget = remaining_time / max(1, ranks_to_try)
        time_budget = min(time_budget, 15.0)
        
        if time_budget < 1.0:
            break
        
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            err = _compute_decomposition_error(U, V, W, T)
            if err < 1e-10:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
