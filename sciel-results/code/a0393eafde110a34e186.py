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

def _als_step(n, m, p, rank, U, V, W, T, max_iters=1000, tol=1e-12):
    """
    Efficient ALS for tensor decomposition.
    T shape: (nm, mp, pn)
    U shape: (nm, rank)
    V shape: (mp, rank)
    W shape: (pn, rank)
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Reshape T for efficient least squares
    # T is (nm, mp, pn)
    # For U update: T_a is (mp, pn). We want T_a = sum_r U_ar * (V_br W_cr)
    # Let K_r = outer(V[:, r], W[:, r]) of shape (mp, pn)
    # Stack K_r into A_u of shape (mp*pn, rank)
    # B_u is T reshaped to (nm, mp*pn)
    # Solve B_u = U @ A_u.T ? No.
    # T_{a, bc} = sum_r U_{ar} K_{bc, r}
    # Let vec(T_a) = B_u[a, :]
    # Let vec(K_r) = A_u[:, r]
    # Then B_u = U @ A_u.T
    # U = B_u @ A_u.T @ inv(A_u @ A_u.T)
    
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1) # T_{b, ac}
    B_w = T.transpose(2, 0, 1).reshape(pn, -1) # T_{c, ab}
    
    err = float('inf')
    for it in range(max_iters):
        # Update U
        # K_r = outer(V[:, r], W[:, r])
        # A_u[:, r] = vec(K_r)
        # A_u = (V @ W.T).reshape(mp, pn, rank).transpose(1, 0, 2).reshape(-1, rank)
        # Wait, V is (mp, rank), W is (pn, rank).
        # V @ W.T is (mp, pn) for each r? No.
        # V[:, r] is (mp, 1). W[:, r] is (pn, 1).
        # outer(V[:, r], W[:, r]) is (mp, pn).
        # We can compute all at once:
        # K = np.einsum('br,cr->bcr', V, W) # (mp, pn, rank)
        # A_u = K.transpose(1, 0, 2).reshape(-1, rank)
        
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            # Solve B_u = U A_u.T
            # U = B_u A_u.T (A_u A_u.T)^-1
            # Use lstsq for stability
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            pass

        # Update V
        # L_r = outer(U[:, r], W[:, r])
        # L is (nm, pn, rank)
        # A_v = vec(L_r)
        # B_v[a, b, c] = T[b, a, c]?
        # T_{a b c} = sum_r V_{br} (U_{ar} W_{cr})
        # For fixed b, T_{a b c} = sum_r V_{br} M_{ac, r}
        # M_{ac, r} = U_{ar} W_{cr}
        # K_v = np.einsum('ar,cr->acr', U, W) # (nm, pn, rank)
        # A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        # B_v should be T reshaped such that row b corresponds to T_{* b *}
        # T is (nm, mp, pn).
        # We want B_v[b, :] to be vec(T_{* b *})
        # T_{a b c}. Indices a, c.
        # So B_v = T.transpose(1, 0, 2).reshape(mp, -1)
        
        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            pass

        # Update W
        # N_r = outer(U[:, r], V[:, r])
        # N is (nm, mp, rank)
        # A_w = vec(N_r)
        # B_w[c, :] = vec(T_{* * c})
        # T_{a b c}. Indices a, b.
        # So B_w = T.transpose(2, 0, 1).reshape(pn, -1)
        
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

def _try_rank_als_fast(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using fast ALS."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        return U, V, W, True
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
    Try to find a lower rank decomposition using fast ALS.
    """
    n, m, p = 2, 3, 10
    
    start_time = time.time()
    
    # Try ranks from 30 to 59
    for rank in range(30, 60):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try a few seeds
        # Higher ranks might converge faster or easier?
        # Let's try 5 seeds for low ranks, 2 for high
        num_seeds = 5 if rank <= 40 else 2
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U, V, W, success = _try_rank_als_fast(n, m, p, rank, max_iters=1000, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the fast ALS approach
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fall back to trivial decomposition
    return _get_trivial_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
