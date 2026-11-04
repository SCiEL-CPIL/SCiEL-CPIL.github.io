import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-8):
    # T_abc = sum_r U_ar V_br W_cr
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid entries
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check invalid entries
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
                
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _als_decompose(rank, max_iters=200, tol=1e-9, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    # Initialize with random values
    U = rng.randn(9, rank)
    V = rng.randn(9, rank)
    W = rng.randn(9, rank)
    
    # Target Tensor M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
                
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Update U
        VW = np.einsum('br,cr->bcr', V, W) # (9, 9, rank)
        B_U = VW.transpose(2, 0, 1).reshape(rank, 81)
        M_mat_U = M.transpose(0, 1, 2).reshape(9, 81)
        
        try:
            U_T, _, _, _ = np.linalg.lstsq(B_U.T, M_mat_U.T, rcond=None)
            U = U_T.T
        except:
            pass
            
        # Update V
        UW = np.einsum('ar,cr->acr', U, W) # (9, 9, rank)
        B_V = UW.transpose(2, 0, 1).reshape(rank, 81)
        M_mat_V = M.transpose(1, 0, 2).reshape(9, 81)
        
        try:
            V_T, _, _, _ = np.linalg.lstsq(B_V.T, M_mat_V.T, rcond=None)
            V = V_T.T
        except:
            pass
            
        # Update W
        UV = np.einsum('ar,br->abr', U, V) # (9, 9, rank)
        B_W = UV.transpose(2, 0, 1).reshape(rank, 81)
        M_mat_W = M.transpose(2, 0, 1).reshape(9, 81)
        
        try:
            W_T, _, _, _ = np.linalg.lstsq(B_W.T, M_mat_W.T, rcond=None)
            W = W_T.T
        except:
            pass
            
        # Calculate Error
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - M)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
    # Snap to half-integers
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def construct():
    # Try rank 23 with more seeds
    for seed in range(100):
        if time.time() > _DEADLINE - 5:
            break
        res = _als_decompose(23, max_iters=100, seed=seed)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 24
    for seed in range(50):
        if time.time() > _DEADLINE - 5:
            break
        res = _als_decompose(24, max_iters=100, seed=seed)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback 27
    U_fb = np.zeros((9, 27), dtype=float)
    V_fb = np.zeros((9, 27), dtype=float)
    W_fb = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_fb[i*3+j, col] = 1.0
                V_fb[j*3+k, col] = 1.0
                W_fb[k*3+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
