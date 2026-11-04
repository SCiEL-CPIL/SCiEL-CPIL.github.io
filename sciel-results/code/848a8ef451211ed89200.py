import time
import numpy as np

def _build_multiplication_tensor():
    """Build the 9x9x9 multiplication tensor for 3x3 matrix mult.
    T[a,b,c] = 1 if a=i*3+j, b=j*3+k, c=k*3+i for some i,j,k.
    """
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i*3+j, j*3+k, k*3+i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W decompose tensor T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _round_to_half_int(X):
    """Round matrix to nearest half-integer."""
    return np.round(2.0 * X) / 2.0

def _is_half_integer(X, tol=1e-6):
    """Check if all entries are close to half-integers."""
    return np.allclose(2.0 * X, np.round(2.0 * X), atol=tol)

def _find_decomposition(T, target_rank, n_trials=5, max_iters=400, tol=1e-10):
    """Find a rank-r decomposition of tensor T using ALS with multiple restarts."""
    n, m, p = T.shape
    r = target_rank
    best_err = float('inf')
    best_UVW = None
    
    for trial in range(n_trials):
        seed = trial * 7919 + 13
        np.random.seed(seed)
        U = np.random.randn(n, r) * 0.1
        V = np.random.randn(m, r) * 0.1
        W = np.random.randn(p, r) * 0.1
        
        for it in range(max_iters):
            # Update U
            M_VW = np.einsum('br,cr->bcr', V, W).reshape(m*p, r)
            T_flat = T.reshape(n, m*p)
            try:
                U = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
            except Exception:
                break
            
            # Update V
            M_UW = np.einsum('ar,cr->acr', U, W).reshape(n*p, r)
            T_flat2 = T.transpose(0, 2, 1).reshape(n*p, m)
            try:
                V = np.linalg.lstsq(M_UW.T, T_flat2.T, rcond=None)[0].T
            except Exception:
                break
            
            # Update W
            M_UV = np.einsum('ar,br->abr', U, V).reshape(n*m, r)
            T_flat3 = T.transpose(2, 0, 1).reshape(p, n*m)
            try:
                W = np.linalg.lstsq(M_UV.T, T_flat3.T, rcond=None)[0].T
            except Exception:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        if err < best_err:
            best_err = err
            best_UVW = (U.copy(), V.copy(), W.copy())
    
    return best_UVW[0], best_UVW[1], best_UVW[2], best_err

def construct():
    """Return a low-rank decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    T = _build_multiplication_tensor()
    
    best_result = None
    for target_rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 15:
            break
        
        n_trials = 5 if target_rank <= 24 else 3
        max_iters = 300 if target_rank <= 24 else 200
        U, V, W, err = _find_decomposition(T, target_rank, n_trials=n_trials, max_iters=max_iters)
        
        if err < 1e-6:
            # Try rounding to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-8):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
            elif _verify_decomposition(U, V, W, T, tol=1e-8) and \
                 _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is None:
        # Fallback: schoolbook rank-27 decomposition
        n, m, p = 3, 3, 3
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
