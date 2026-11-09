import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(T, U, V, W, tol=1e-6):
    r = U.shape[1]
    # Reconstruct tensor
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Use einsum for speed
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    residual = T - recon
    return np.max(np.abs(residual)) < tol

def _als_step(T, U, V, W, rank, max_iter=50, tol=1e-9):
    """
    Perform ALS iterations.
    """
    a_size = T.shape[0]
    b_size = T.shape[1]
    c_size = T.shape[2]
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        # T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]
        # Let M_U[b,c,r] = V[b,r] W[c,r]
        # T_flat[a, b*c] = sum_r U[a,r] M_U[b,c,r]
        # T_flat = U @ M_U_reshaped.T
        M_U = np.einsum('br,cr->bcr', V, W).reshape(b_size * c_size, rank)
        T_flat = T.reshape(a_size, b_size * c_size)
        # Solve U = T_flat @ pinv(M_U.T)
        # Using lstsq for stability
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        M_V = np.einsum('ar,cr->acr', U, W).reshape(a_size * c_size, rank)
        T_flat_v = T.reshape(a_size, c_size, b_size).transpose(0, 2, 1).reshape(b_size, a_size * c_size)
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        M_W = np.einsum('ar,br->abr', U, V).reshape(a_size * b_size, rank)
        T_flat_w = T.reshape(a_size, b_size, c_size).transpose(2, 0, 1).reshape(c_size, a_size * b_size)
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        # Check convergence
        # err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W), 'fro')
        # if abs(prev_err - err) < tol:
        #     break
        # prev_err = err
        
    return U, V, W

def _try_rank(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition.
    """
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    start_time = time.time()
    
    # Number of restarts based on time limit
    # Each restart takes ~10ms to 50ms for small tensors
    # Let's try up to 100 restarts or until time limit
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        # Scale matters for ALS
        U = np.random.randn(a_size, rank) * 0.1
        V = np.random.randn(b_size, rank) * 0.1
        W = np.random.randn(c_size, rank) * 0.1
        
        U, V, W = _als_step(T, U, V, W, rank, max_iter=20)
        
        if _verify_exact(T, U, V, W, tol=1e-5):
            # Polish with higher precision
            U, V, W = _als_step(T, U, V, W, rank, max_iter=50)
            if _verify_exact(T, U, V, W, tol=1e-8):
                return U, V, W
                
    return None

def _to_list(mat):
    result = []
    for row in mat:
        row_list = []
        for val in row:
            # Check for integer or half-integer
            nearest_int = round(val)
            if abs(val - nearest_int) < 1e-5:
                row_list.append(int(nearest_int))
            else:
                nearest_half = round(2 * val) / 2.0
                if abs(val - nearest_half) < 1e-5:
                    row_list.append(nearest_half)
                else:
                    row_list.append(float(val))
        result.append(row_list)
    return result

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p # 48
    
    time_start = time.time()
    time_limit_total = 140 # 10s margin
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Try ranks from a lower bound up to schoolbook - 1
    # Lower bound is 24. 
    # We suspect the rank might be 32 or 36.
    # We will try ranks in descending order of likelihood or just scan up?
    # Scanning up from 24 is safer for "minimal" if we find one, but might waste time on impossible ranks.
    # However, 24-31 are likely impossible.
    # Let's try 32, 34, 36, 38, 40, 42, 44, 46.
    # Or just scan 24 to 47 with a very tight time limit per rank.
    
    # Given the small size, we can afford to try more ranks.
    # Time per rank: 5 seconds.
    # 48 - 24 = 24 ranks. 24 * 5 = 120s. Too much.
    # Let's prioritize ranks that are multiples of 2 or 4, as tensor ranks often have structure.
    # Or just try 32, 30, 28, 26, 24 first. If none work, try 34, 36...
    
    candidate_ranks = [32, 30, 28, 26, 24, 34, 36, 38, 40, 42, 44, 46]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
            
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        # Allocate min(5s, remaining/2) to each rank
        per_try_time = min(5.0, remaining_time / 2.0)
        
        if per_try_time < 0.5:
            break
            
        result = _try_rank(T, n, m, p, rank, per_try_time, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break # Found a valid decomposition, return it
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Fall back to schoolbook
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
