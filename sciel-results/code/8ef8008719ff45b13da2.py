"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, max_iters=500, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None, polish_iters=100, polish_tol=1e-12):
    """
    Alternating Least Squares for tensor decomposition with polish phase.
    T: target tensor (dim_a, dim_b, dim_c)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = T.shape
    
    if init_U is not None:
        U = init_U.copy()
    else:
        U = np.random.randn(dim_a, rank) * 0.1
    
    if init_V is not None:
        V = init_V.copy()
    else:
        V = np.random.randn(dim_b, rank) * 0.1
    
    if init_W is not None:
        W = init_W.copy()
    else:
        W = np.random.randn(dim_c, rank) * 0.1
    
    prev_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            U[a, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
        
        # Update V
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            V[b, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
        
        # Update W
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            W[c, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
        
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                break
        prev_error = error
    
    # Polish phase: tighter tolerance, more iterations on best solution
    if best_error < 1e-3 and (deadline is None or time.time() < deadline - 5):
        U, V, W = best_U, best_V, best_W
        prev_error = float('inf')
        for it in range(polish_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
            for a in range(dim_a):
                T_flat = T[a, :, :].reshape(-1)
                U[a, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            
            M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
            for b in range(dim_b):
                T_flat = T[:, b, :].reshape(-1)
                V[b, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            
            M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
            for c in range(dim_c):
                T_flat = T[:, :, c].reshape(-1)
                W[c, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_approx - T)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_error - error) < polish_tol:
                break
            prev_error = error
        
        if best_error < 1e-9:
            return best_U, best_V, best_W
    
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-9:
        return best_U, best_V, best_W
    
    return None

def _try_rank23(n, m, p, deadline, max_attempts=300):
    """
    Try to find a rank-23 decomposition using ALS with multiple initializations.
    Uses varied perturbation scales and drop patterns to escape local minima.
    """
    T = _build_target_tensor(n, m, p)
    target_rank = 23
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n * m * p
    n_drop = total_terms - target_rank
    
    all_indices = list(range(total_terms))
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        rng = random.Random(42 + attempt * 7)
        
        # Vary the perturbation scale more aggressively
        # Use different scales to explore different regions of the solution space
        if attempt % 3 == 0:
            perturb_scale = rng.uniform(0.001, 0.01)
        elif attempt % 3 == 1:
            perturb_scale = rng.uniform(0.01, 0.05)
        else:
            perturb_scale = rng.uniform(0.05, 0.1)
        
        # Pick indices to drop - use different strategies
        if attempt % 5 == 0:
            drop_indices = rng.sample(all_indices, n_drop)
        elif attempt % 5 == 1:
            # Drop consecutive indices
            start = rng.randint(0, total_terms - n_drop)
            drop_indices = list(range(start, start + n_drop))
        elif attempt % 5 == 2:
            # Drop every kth index
            k = max(1, total_terms // n_drop)
            drop_indices = list(range(0, total_terms, k))[:n_drop]
        elif attempt % 5 == 3:
            # Drop from beginning
            drop_indices = list(range(n_drop))
        else:
            # Drop from end
            drop_indices = list(range(total_terms - n_drop, total_terms))
        
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices].copy()
        init_V = V0[:, keep_indices].copy()
        init_W = W0[:, keep_indices].copy()
        
        # Add random perturbation to break symmetry
        for r_idx in range(len(keep_indices)):
            mask_u = init_U[:, r_idx] != 0
            mask_v = init_V[:, r_idx] != 0
            mask_w = init_W[:, r_idx] != 0
            if mask_u.any():
                init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_v.any():
                init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_w.any():
                init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
        
        # Run ALS with adaptive max_iters based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        max_iters = min(200, max(50, int(remaining * 2)))
        
        result = _als_decompose(T, target_rank, max_iters=max_iters, tol=1e-11, 
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # Verify with tighter tolerance
            if _verify_fast(U, V, W, n, m, p, tol=1e-10):
                return U, V, W
    
    return None

def _try_rank23_direct(n, m, p, deadline):
    """
    Try a direct construction of rank-23 decomposition with varied random initializations.
    """
    T = _build_target_tensor(n, m, p)
    target_rank = 23
    
    # Try with different random seeds and initialization scales
    scales = [0.1, 0.3, 0.5, 1.0, 2.0]
    for seed in range(80):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(seed)
        scale = scales[seed % len(scales)]
        
        # Initialize with random values of varying scale
        U = rng.randn(9, target_rank) * scale
        V = rng.randn(9, target_rank) * scale
        W = rng.randn(9, target_rank) * scale
        
        # Run ALS
        result = _als_decompose(T, target_rank, max_iters=200, tol=1e-11,
                                init_U=U, init_V=V, init_W=W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-10):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin from 150s limit
    
    # First, try to find a rank-23 decomposition via ALS with schoolbook initialization
    result = _try_rank23(n, m, p, deadline, max_attempts=250)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try direct random initialization
    if time.time() < deadline - 10:
        result = _try_rank23_direct(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
