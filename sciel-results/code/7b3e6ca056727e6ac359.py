import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 11]

# ... existing helper functions ...

def _find_low_rank_decomposition(n, m, p, target_rank, deadline):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    for attempt in range(50): # Increased restarts
        if time.time() > deadline - 15:
            break
            
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        def fun(x):
            Uc = x[:n*m*target_rank].reshape(n*m, target_rank)
            Vc = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            Wc = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            T_recon = np.einsum('ar,br,cr->abc', Uc, Vc, Wc)
            return (T_recon - T).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(fun, x0, method='lm', max_nfev=2000)
            err = np.linalg.norm(result.fun)
            if err < 1e-6:
                Uc = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                Vc = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                Wc = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Try to round to half-integers
                # Check if 2*Uc is close to integer
                def try_round(arr):
                    scaled = arr * 2
                    rounded = np.round(scaled)
                    if np.allclose(scaled, rounded, atol=1e-4):
                        return rounded / 2.0
                    return None
                
                Ur = try_round(Uc)
                Vr = try_round(Vc)
                Wr = try_round(Wc)
                
                if Ur is not None and Vr is not None and Wr is not None:
                    # Verify exactness with rounded values
                    T_recon_r = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
                    if np.allclose(T_recon_r, T, atol=1e-9):
                        if err < best_err:
                            best_err = err
                            best_U = Ur
                            best_V = Vr
                            best_W = Wr
                            if err < 1e-9:
                                break
        except Exception:
            continue
            
    if best_U is not None:
        return best_U, best_V, best_W
    return None

def _build_rank37(deadline):
    n, m, p = DIMENSIONS
    
    U_block, V_block, W_block = _find_low_rank_decomposition(2, 2, 3, 10, deadline)
    
    if U_block is None:
        return None
        
    T_block = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T_block[i*2+j, j*3+k, k*2+i] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U_block, V_block, W_block)
    if not np.allclose(T_recon, T_block, atol=1e-6):
        return None
        
    rank_total = 10 * 3 + 7
    U = np.zeros((4, rank_total))
    V = np.zeros((22, rank_total))
    W = np.zeros((22, rank_total))
    
    r = 0
    for block_idx in range(3):
        col_offset = block_idx * 3
        
        V_block_mapped = np.zeros((22, 10))
        W_block_mapped = np.zeros((22, 10))
        
        for j in range(2):
            for k in range(3):
                v_idx_local = j*3 + k
                v_idx_global = j*11 + (col_offset + k)
                V_block_mapped[v_idx_global, :] = V_block[v_idx_local, :]
                
        for k in range(3):
            for i in range(2):
                w_idx_local = k*2 + i
                w_idx_global = (col_offset + k)*2 + i
                W_block_mapped[w_idx_global, :] = W_block[w_idx_local, :]
                
        U[:, r:r+10] = U_block
        V[:, r:r+10] = V_block_mapped
        W[:, r:r+10] = W_block_mapped
        r += 10
        
    k0, k1 = 9, 10
    v_e = k0
    v_f = k1
    v_g = 11 + k0
    v_h = 11 + k1
    w_c00 = 2 * k0
    w_c10 = 2 * k0 + 1
    w_c01 = 2 * k1
    w_c11 = 2 * k1 + 1
    
    U[0, r] += 1; U[3, r] += 1
    V[v_e, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1; W[w_c11, r] += 1
    r += 1
    
    U[2, r] += 1; U[3, r] += 1
    V[v_e, r] += 1
    W[w_c10, r] += 1; W[w_c11, r] += -1
    r += 1
    
    U[0, r] += 1
    V[v_f, r] += 1; V[v_h, r] += -1
    W[w_c01, r] += 1; W[w_c11, r] += 1
    r += 1
    
    U[3, r] += 1
    V[v_g, r] += 1; V[v_e, r] += -1
    W[w_c00, r] += 1; W[w_c10, r] += 1
    r += 1
    
    U[0, r] += 1; U[1, r] += 1
    V[v_h, r] += 1
    W[w_c00, r] += -1; W[w_c01, r] += 1
    r += 1
    
    U[2, r] += 1; U[0, r] += -1
    V[v_e, r] += 1; V[v_f, r] += 1
    W[w_c11, r] += 1
    r += 1
    
    U[1, r] += 1; U[3, r] += -1
    V[v_g, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1
    r += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    res37 = _build_rank37(deadline)
    if res37 is not None:
        U, V, W = res37
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    U, V, W = _build_rank39()
    T = _build_tensor()
    
    if not _verify(U, V, W, T):
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 39

    if time.time() < deadline - 15:
        res = _reduce_rank_svd(U, V, W, deadline, T)
        if res is not None:
            U_r, V_r, W_r, rk = res
            if _verify(U_r, V_r, W_r, T):
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rk

    if time.time() < deadline - 5:
        res2 = _rank1_completion_search(best_U, best_V, best_W, deadline, T)
        if res2 is not None:
            U_c, V_c, W_c, rk2 = res2
            if _verify(U_c, V_c, W_c, T) and rk2 < best_rank:
                best_U, best_V, best_W, best_rank = U_c, V_c, W_c, rk2

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
