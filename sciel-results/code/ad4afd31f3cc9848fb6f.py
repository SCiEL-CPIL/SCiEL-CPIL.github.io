import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _als_step(n, m, p, rank, U, V, W, T, max_iters=2000, tol=1e-12):
    """
    Efficient ALS for tensor decomposition.
    T shape: (nm, mp, pn)
    U shape: (nm, rank)
    V shape: (mp, rank)
    W shape: (pn, rank)
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Pre-compute reshaped tensors for efficiency
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1)
    B_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    err = float('inf')
    for it in range(max_iters):
        # Update U
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            pass

        # Update V
        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            pass

        # Update W
        K_w = np.einsum('ar,br->abr', U, V)
        A_w = K_w.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            W = np.linalg.lstsq(A_w.T, B_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
            
    return U, V, W, err

def _polish_decomposition(U, V, W, T, max_iters=5000, tol=1e-14):
    """High-precision polish pass with tighter tolerance."""
    nm, mp, pn = T.shape
    rank = U.shape[1]
    
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1)
    B_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    err = float('inf')
    for it in range(max_iters):
        # Update U
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            break

        # Update V
        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            break

        # Update W
        K_w = np.einsum('ar,br->abr', U, V)
        A_w = K_w.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            W = np.linalg.lstsq(A_w.T, B_w.T, rcond=None)[0].T
        except:
            break
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
            
    return U, V, W, err

def _try_rank_als_fast(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using fast ALS."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        # Polish for higher precision
        U, V, W, err_polished = _polish_decomposition(U, V, W, T, max_iters=2000, tol=1e-14)
        if err_polished < 1e-10:
            return U, V, W, True
    
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _try_rank_with_basis_transform(n, m, p, rank, max_iters=2000, seed=42):
    """
    Try rank-r decomposition after applying a linear change of basis to the output space.
    This exploits the structure where the 10 output columns (k dimension) can be treated
    as a vector space, potentially revealing low-rank structure in matricized views.
    """
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    pn = p * n  # 20
    
    # Apply random orthogonal transformation to output space
    # This changes the basis of the 20-dimensional output space
    Q = rng.randn(pn, pn)
    Q, _ = np.linalg.qr(Q)
    
    # Transform tensor: T'[a,b,c'] = sum_c T[a,b,c] * Q[c',c]
    T_transformed = np.einsum('abc,dc->abd', T, Q)
    
    # Run ALS on transformed tensor
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(pn, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T_transformed, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        # Transform back: W_original = W_transformed @ Q.T
        W_original = W @ Q.T
        
        # Verify against original tensor
        T_recon = np.einsum('ar,br,cr->abc', U, V, W_original)
        err_orig = np.max(np.abs(T_recon - T))
        
        if err_orig < 1e-10:
            # Polish
            U, V, W_original, err_polished = _polish_decomposition(U, V, W_original, T, max_iters=2000, tol=1e-14)
            if err_polished < 1e-10:
                return U, V, W_original, True
        
        if err_orig < 1e-10:
            return U, V, W_original, True
    
    return U, V, W, False

def _get_trivial_decomposition(n, m, p):
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(n, m, p):
    """
    Try to find a lower rank decomposition using fast ALS with basis transformations.
    """
    n, m, p = 2, 3, 10
    
    start_time = time.time()
    
    # Try ranks from 55 to 59 first (most likely to be lower if possible)
    # Then 50-54, then lower
    ranks_to_try = list(range(55, 60)) + list(range(50, 55)) + list(range(40, 50))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try more seeds for higher ranks (closer to 60)
        num_seeds = 10 if rank >= 55 else 5
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # First try without basis transform
            U, V, W, success = _try_rank_als_fast(n, m, p, rank, max_iters=1500, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try with basis transformation to output space
            U, V, W, success = _try_rank_with_basis_transform(n, m, p, rank, max_iters=1500, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the fast ALS approach with basis transformations
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fall back to trivial decomposition
    return _get_trivial_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
