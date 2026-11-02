import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0
    
    def _snap_to_half_integer(arr, tol=1e-6):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_computed, T_target, atol=tol)

    def _try_rank(n, m, p, target_rank, max_iters=20, seed=42):
        rng = np.random.default_rng(seed)
        U = rng.normal(0, 0.1, (n*m, target_rank))
        V = rng.normal(0, 0.1, (m*p, target_rank))
        W = rng.normal(0, 0.1, (p*n, target_rank))
        
        def compute_error(U, V, W):
            U_r = U.reshape(n, m, -1)
            V_r = V.reshape(m, p, -1)
            W_r = W.reshape(p, n, -1)
            T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            T_target = np.ones((n, m, p))
            return np.sum((T_comp - T_target)**2)
        
        from scipy.optimize import minimize
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            return compute_error(U_flat, V_flat, W_flat)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < 1e-10:
                U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                U_snap = _snap_to_half_integer(U_sol)
                V_snap = _snap_to_half_integer(V_sol)
                W_snap = _snap_to_half_integer(W_sol)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _verify_decomposition(n, m, p, U_snap, V_snap, W_snap, tol=1e-9):
                        return U_snap, V_snap, W_snap
        except Exception:
            pass
        return None

    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
    
    # Try to find blocks
    block_233 = _try_rank(2, 3, 3, 16)
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    if block_233 is None:
        block_233 = _try_rank(2, 3, 3, 17)
        
    if time.time() > deadline:
        return _schoolbook(n, m, p)
        
    block_223 = _try_rank(2, 2, 3, 11)
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    if block_223 is None:
        block_223 = _try_rank(2, 2, 3, 12)

    if block_233 is None or block_223 is None:
        return _schoolbook(n, m, p)
        
    R_223 = block_223[0].shape[1]
    R_233 = block_233[0].shape[1]
    Total_Rank = 2 * R_223 + 2 * R_233
    
    U_full = np.zeros((2*5, Total_Rank))
    V_full = np.zeros((5*6, Total_Rank))
    W_full = np.zeros((6*2, Total_Rank))
    
    def place_block_223(U_b, V_b, W_b, j_offset, k_offset, rank_offset):
        for i in range(2):
            for j_b in range(2):
                j_global = j_offset + j_b
                U_full[i * 5 + j_global, rank_offset:rank_offset+R_223] = U_b[i * 2 + j_b, :]
        for j_b in range(2):
            j_global = j_offset + j_b
            for k_b in range(3):
                k_global = k_offset + k_b
                V_full[j_global * 6 + k_global, rank_offset:rank_offset+R_223] = V_b[j_b * 3 + k_b, :]
        for k_b in range(3):
            k_global = k_offset + k_b
            for i in range(2):
                W_full[k_global * 2 + i, rank_offset:rank_offset+R_223] = W_b[k_b * 2 + i, :]
                
    def place_block_233(U_b, V_b, W_b, j_offset, k_offset, rank_offset):
        for i in range(2):
            for j_b in range(3):
                j_global = j_offset + j_b
                U_full[i * 5 + j_global, rank_offset:rank_offset+R_233] = U_b[i * 3 + j_b, :]
        for j_b in range(3):
            j_global = j_offset + j_b
            for k_b in range(3):
                k_global = k_offset + k_b
                V_full[j_global * 6 + k_global, rank_offset:rank_offset+R_233] = V_b[j_b * 3 + k_b, :]
        for k_b in range(3):
            k_global = k_offset + k_b
            for i in range(2):
                W_full[k_global * 2 + i, rank_offset:rank_offset+R_233] = W_b[k_b * 2 + i, :]
                
    place_block_223(block_223[0], block_223[1], block_223[2], 0, 0, 0)
    place_block_223(block_223[0], block_223[1], block_223[2], 0, 3, R_223)
    place_block_233(block_233[0], block_233[1], block_233[2], 2, 0, 2*R_223)
    place_block_233(block_233[0], block_233[1], block_233[2], 2, 3, 2*R_223 + R_233)
    
    if _verify_decomposition(n, m, p, U_full, V_full, W_full, tol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
