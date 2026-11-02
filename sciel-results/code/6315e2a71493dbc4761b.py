import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.max(np.abs(recon - T)) < tol

def _construct_trivial(n, m, p):
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

def _round_to_half_int(M, tol=1e-4):
    """Try to round matrix to nearest half-integers. Returns (rounded, success)."""
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(n, m, p, R, max_iter=100, tol=1e-7, seed=42):
    """
    Alternating least squares for 3-tensor decomposition of matrix multiplication tensor.
    Finds U (nm x R), V (mp x R), W (pn x R) such that
    sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] where T[i*m+j, j*p+k, k*n+i] = 1.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Random initialization
    U = np.random.randn(nm, R) * 0.5
    V = np.random.randn(mp, R) * 0.5
    W = np.random.randn(pn, R) * 0.5
    
    for it in range(max_iter):
        # Update U: for each row a, solve least squares
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp*pn, R)
        MW_T = MW_flat.T  # (R, mp*pn)
        
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            # Solve MW_T @ x = target_a in least squares sense
            x, _, _, _ = np.linalg.lstsq(MW_T, target_a, rcond=None)
            U[a, :] = x
        
        # Update V: for each row b, solve least squares
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm*pn, R)
        UW_T = UW_flat.T  # (R, nm*pn)
        
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            x, _, _, _ = np.linalg.lstsq(UW_T, target_b, rcond=None)
            V[b, :] = x
        
        # Update W: for each row c, solve least squares
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm*mp, R)
        UV_T = UV_flat.T  # (R, nm*mp)
        
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            x, _, _, _ = np.linalg.lstsq(UV_T, target_c, rcond=None)
            W[c, :] = x
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    
    if err < 1e-4:
        # Try to round to half-integers
        U_r, ok1 = _round_to_half_int(U, tol=1e-3)
        if ok1:
            V_r, ok2 = _round_to_half_int(V, tol=1e-3)
            if ok2:
                W_r, ok3 = _round_to_half_int(W, tol=1e-3)
                if ok3:
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                        return U_r, V_r, W_r, 0.0
                    # Try tighter rounding
                    U_r2, ok1b = _round_to_half_int(U, tol=1e-5)
                    if ok1b:
                        V_r2, ok2b = _round_to_half_int(V, tol=1e-5)
                        if ok2b:
                            W_r2, ok3b = _round_to_half_int(W, tol=1e-5)
                            if ok3b:
                                if _verify_decomposition(U_r2, V_r2, W_r2, n, m, p, tol=1e-6):
                                    return U_r2, V_r2, W_r2, 0.0
    
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with trivial decomposition
    best_result = _construct_trivial(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But tensor rank is typically higher. Let's try from 30 up to 89.
    for R in range(30, 90):
        if time.time() > deadline - 15.0:
            break
        
        # Try several random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            seed = 42 + restart * 1000 + R
            result = _als_decompose(n, m, p, R, max_iter=80, tol=1e-6, seed=seed)
            if result is not None:
                U_r, V_r, W_r, err = result
                # Convert to lists
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
