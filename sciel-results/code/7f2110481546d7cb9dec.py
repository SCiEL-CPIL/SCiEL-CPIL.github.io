import numpy as np
import time

def _construct_schoolbook(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _als_decompose(T, rank, n, m, p, max_iter=100, tol=1e-10, seed=42, time_limit=None):
    """
    Alternating least squares for 3-tensor decomposition.
    T has shape (n*m, m*p, p*n)
    Returns (U, V, W) or None if not converged.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    np.random.seed(seed)
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        # Update U: fix V, W. For each row a of U, solve:
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
        # This is: U[a,:] @ M = T[a].flat where M[b,c,r] = V[b,r]*W[c,r]
        # M_flat has shape (mp*pn, rank), so we solve M_flat.T @ U[a,:] = T[a].flat
        
        # Build M_flat: M[b,c,r] = V[b,r] * W[c,r]
        # Efficiently: M_flat = (V * W[None,:,:]).reshape(mp*pn, rank)
        # Actually: M[b,c,r] = V[b,r] * W[c,r]
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, rank)
        MW_flat = MW.reshape(mp * pn, rank)
        
        # For each a, solve: MW_flat.T @ U[a,:] = T[a].reshape(-1)
        # This is a least squares problem: (rank x mp*pn) @ (rank,) = (mp*pn,)
        for a in range(nm):
            target = T[a].reshape(-1)
            # Solve MW_flat.T @ x = target in least squares
            # MW_flat.T has shape (rank, mp*pn)
            U[a, :], residuals, _, _ = np.linalg.lstsq(MW_flat.T, target, rcond=None)
        
        # Update V: fix U, W. For each row b of V, solve:
        # sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c] for all a,c
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, rank)
        UW_flat = UW.reshape(nm * pn, rank)
        
        for b in range(mp):
            target = T[:, b, :].reshape(-1)
            V[b, :], residuals, _, _ = np.linalg.lstsq(UW_flat.T, target, rcond=None)
        
        # Update W: fix U, V. For each row c of W, solve:
        # sum_r W[c,r] * U[a,r] * V[b,r] = T[a,b,c] for all a,b
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, rank)
        UV_flat = UV.reshape(nm * mp, rank)
        
        for c in range(pn):
            target = T[:, :, c].reshape(-1)
            W[c, :], residuals, _, _ = np.linalg.lstsq(UV_flat.T, target, rcond=None)
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
        if err > prev_err * 1.1:
            # Diverging, break
            break
        prev_err = err
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        return U, V, W
    return None

def _try_rank(T, R, n, m, p, n_restarts=5, max_iter=80, time_limit=None):
    """Try to find a decomposition of rank R."""
    start_time = time.time()
    for restart in range(n_restarts):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        seed = 42 + restart * 1000
        result = _als_decompose(T, R, n, m, p, max_iter=max_iter, tol=1e-10, seed=seed, 
                                time_limit=time_limit)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin from 150s limit
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But tensor rank can be higher. Try from 30 up to 90.
    best_result = None
    
    for R in range(30, 91):
        if time.time() > deadline - 15:
            break
        
        # Decrease number of restarts as rank increases (higher ranks are easier)
        if R <= 40:
            n_restarts = 3
            max_iter = 60
        elif R <= 60:
            n_restarts = 3
            max_iter = 50
        else:
            n_restarts = 2
            max_iter = 40
        
        result = _try_rank(T, R, n, m, p, n_restarts=n_restarts, max_iter=max_iter,
                           time_limit=deadline - 10)
        if result is not None:
            U, V, W = result
            # Convert to lists with integer/half-integer rounding
            def _to_clean_list(M):
                M = np.asarray(M, dtype=float)
                # Try rounding to half-integers
                M2 = 2.0 * M
                Mr = np.round(M2)
                if np.max(np.abs(M2 - Mr)) < 1e-4:
                    M = Mr / 2.0
                    # Convert to int where possible
                    out = []
                    for row in M:
                        out_row = []
                        for val in row:
                            if abs(val - round(val)) < 1e-9:
                                out_row.append(int(round(val)))
                            else:
                                out_row.append(round(val * 2) / 2.0)
                        out.append(out_row)
                    return out
                else:
                    # Keep as float
                    return M.tolist()
            
            return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
    
    # Fallback to schoolbook
    result = _construct_schoolbook(n, m, p)
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
