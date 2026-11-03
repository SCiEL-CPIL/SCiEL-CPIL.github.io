import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _khatri_rao(A, B):
    """Compute Khatri-Rao product of two matrices A and B."""
    # A is (n, r), B is (m, r)
    # Result is (n*m, r) where result[i*m + j, r] = A[i, r] * B[j, r]
    return np.einsum('ir,jr->ijr', A, B).reshape(A.shape[0] * B.shape[0], -1)

def _als_decompose(T, rank, max_iter=100, tol=1e-12, seed=0):
    """
    Alternating Least Squares to decompose T into U, V, W.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    # Initialize factors with random values
    # Scale to avoid large numbers
    U = rng.randn(a, rank) * 0.5
    V = rng.randn(b, rank) * 0.5
    W = rng.randn(c, rank) * 0.5
    
    T_ac = T.reshape(a, b * c)
    T_bc = T.transpose(1, 0, 2).reshape(b, a * c)
    T_cc = T.transpose(2, 0, 1).reshape(c, a * b)
    
    prev_residual = float('inf')
    
    for it in range(max_iter):
        # Update U
        Khatri_VW = _khatri_rao(V, W)
        try:
            U_new = np.linalg.lstsq(Khatri_VW, T_ac.T, rcond=None)[0].T
        except:
            U_new = U
            
        # Update V
        Khatri_UW = _khatri_rao(U_new, W)
        try:
            V_new = np.linalg.lstsq(Khatri_UW, T_bc.T, rcond=None)[0].T
        except:
            V_new = V
            
        # Update W
        Khatri_UV = _khatri_rao(U_new, V_new)
        try:
            W_new = np.linalg.lstsq(Khatri_UV, T_cc.T, rcond=None)[0].T
        except:
            W_new = W
            
        U, V, W = U_new, V_new, W_new
        
        # Compute residual
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
        
        if abs(prev_residual - residual) < tol:
            break
        prev_residual = residual
        
    return U, V, W, residual

def _polish(U, V, W, T, max_nfev=200):
    """
    Use scipy's least_squares for high-precision polish.
    """
    a, b, c = T.shape
    rank = U.shape[1]
    
    T_flat = T.flatten()
    
    def residual_func(params):
        U = params[:a*rank].reshape(a, rank)
        V = params[a*rank:a*rank+b*rank].reshape(b, rank)
        W = params[a*rank+b*rank:].reshape(c, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T).flatten()
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        from scipy.optimize import least_squares
        res = least_squares(residual_func, params, method='lm', max_nfev=max_nfev)
        params = res.x
    except ImportError:
        return U, V, W, np.linalg.norm(residual_func(params)) / np.linalg.norm(T)
    except Exception:
        return U, V, W, np.linalg.norm(residual_func(params)) / np.linalg.norm(T)
        
    U = params[:a*rank].reshape(a, rank)
    V = params[a*rank:a*rank+b*rank].reshape(b, rank)
    W = params[a*rank+b*rank:].reshape(c, rank)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Schoolbook decomposition as fallback
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
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Theoretical lower bound is 27
    # We try ranks from 27 to 53 (if 54 is schoolbook, we look for < 54)
    # If we find 54, it's not an improvement, so we stop early if we can't go lower.
    # However, if the true rank is 54, we should return 54.
    # If the true rank is < 54, we find it.
    
    target_ranks = list(range(27, rank_schoolbook)) # 27 to 53
    
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        best_local_residual = float('inf')
        best_local_U = best_local_V = best_local_W = None
        
        # More restarts for lower ranks
        max_restarts = 20 if r < 35 else 10
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                seed = r * 1000 + attempt
                
                U, V, W, residual = _als_decompose(T, r, max_iter=50, tol=1e-10, seed=seed)
                
                if residual < best_local_residual:
                    best_local_residual = residual
                    best_local_U = U
                    best_local_V = V
                    best_local_W = W
                    
                # If residual is small, try to polish
                if residual < 1e-3:
                    U, V, W, residual = _polish(U, V, W, T, max_nfev=100)
                    
                    if residual < best_local_residual:
                        best_local_residual = residual
                        best_local_U = U
                        best_local_V = V
                        best_local_W = W
                        
                    # If residual is very small, accept
                    if residual < 1e-8:
                        found = True
                        break
                        
            except Exception as e:
                continue
        
        if found and best_local_residual < 1e-8:
            best_result = {
                "U": best_local_U.tolist(),
                "V": best_local_V.tolist(),
                "W": best_local_W.tolist()
            }
            best_rank = r
            break
    
    # If no improvement found, return schoolbook
    if best_result is None:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    else:
        return best_result
