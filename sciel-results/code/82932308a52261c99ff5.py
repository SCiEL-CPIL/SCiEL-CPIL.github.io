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

def _als_decompose(T, rank, max_iters=500, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
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
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            if np.linalg.matrix_rank(M) < rank:
                U[a, :] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            else:
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
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                break
        prev_error = error
    
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    
    return None

def _try_rank23(n, m, p, deadline, max_attempts=300):
    """
    Try to find a rank-23 decomposition using ALS with multiple initializations.
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
        
        rng = random.Random(42 + attempt)
        
        # Pick indices to drop
        drop_indices = rng.sample(all_indices, n_drop)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices].copy()
        init_V = V0[:, keep_indices].copy()
        init_W = W0[:, keep_indices].copy()
        
        # Add small random perturbation to break symmetry
        perturb_scale = rng.uniform(0.001, 0.05)
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
        
        # Run ALS
        result = _als_decompose(T, target_rank, max_iters=200, tol=1e-11, 
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
    
    return None

def _try_rank23_direct(n, m, p, deadline):
    """
    Try a direct construction of rank-23 decomposition.
    This uses a known structure for 3x3 matrix multiplication.
    """
    # For 3x3, we can use the following approach:
    # The tensor has 27 non-zero entries. We need to find 23 rank-1 tensors
    # whose sum equals the target tensor.
    
    # One approach: use the fact that certain combinations of terms can be merged.
    # Let's try to construct it explicitly.
    
    # Actually, let's try a different numerical approach:
    # Use a global optimization with multiple restarts
    
    T = _build_target_tensor(n, m, p)
    target_rank = 23
    
    # Try with different random seeds and initializations
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(9, target_rank) * 0.5
        V = rng.randn(9, target_rank) * 0.5
        W = rng.randn(9, target_rank) * 0.5
        
        # Run ALS
        result = _als_decompose(T, target_rank, max_iters=300, tol=1e-11,
                                init_U=U, init_V=V, init_W=W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # First, try to find a rank-23 decomposition via ALS with schoolbook initialization
    result = _try_rank23(n, m, p, deadline, max_attempts=200)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try direct random initialization
    if time.time() < deadline:
        result = _try_rank23_direct(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
