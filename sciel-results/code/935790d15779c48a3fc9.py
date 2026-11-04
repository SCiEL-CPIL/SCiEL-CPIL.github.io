import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_tensor_333():
    n, m, p = 3, 3, 3
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _als_step_333(U, V, W, T, n, m, p):
    rank = U.shape[1]
    # Update U
    for a in range(n * m):
        target = T[a].flatten()
        design = np.zeros((m * p * p * n, rank))
        for d in range(rank):
            design[:, d] = np.kron(V[:, d], W[:, d])
        sol, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        U[a, :] = sol
    
    # Update V
    for b in range(m * p):
        target = T[:, b, :].flatten()
        design = np.zeros((n * m * p * n, rank))
        for d in range(rank):
            design[:, d] = np.kron(U[:, d], W[:, d])
        sol, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        V[b, :] = sol
    
    # Update W
    for c in range(p * n):
        target = T[:, :, c].flatten()
        design = np.zeros((n * m * m * p, rank))
        for d in range(rank):
            design[:, d] = np.kron(U[:, d], V[:, d])
        sol, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        W[c, :] = sol
    
    return U, V, W

def _search_rank_333(T, max_rank, n, m, p, time_limit, n_restarts=20, n_iters=100):
    start_time = time.time()
    best_result = None
    for r in range(18, max_rank + 1):
        if time.time() - start_time > time_limit * 0.7:
            break
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit * 0.8:
                break
            np.random.seed(42 + r * 1000 + restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(n_iters):
                if time.time() - start_time > time_limit * 0.9:
                    break
                U, V, W = _als_step_333(U, V, W, T, n, m, p)
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-9:
                    best_result = (r, U.copy(), V.copy(), W.copy())
                    return best_result
            
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-9:
                best_result = (r, U.copy(), V.copy(), W.copy())
                return best_result
    return best_result

def construct():
    start_time = time.time()
    time_limit = 140.0
    n, m, p = DIMENSIONS
    
    # Try to find rank 23 for 3x3x3
    T_333 = _build_tensor_333()
    result_333 = _search_rank_333(T_333, 23, 3, 3, 3, time_limit * 0.5)
    
    if result_333 is not None:
        r, U_333, V_333, W_333 = result_333
        # Assemble 3x3x6 solution
        rank_total = 2 * r
        U_total = np.zeros((n * m, rank_total))
        V_total = np.zeros((m * p, rank_total))
        W_total = np.zeros((p * n, rank_total))
        
        # First half: columns 0..r-1
        U_total[:, :r] = U_333
        
        # V mapping:
        # V_333 has shape (9, r). Indices (j, k_local) -> j*3+k_local.
        # V_total has shape (18, rank_total). Indices (j, k) -> j*6+k.
        # For k in 0..2: V_total[j*6+k, d] = V_333[j*3+k, d]
        for j in range(3):
            for k in range(3):
                V_total[j*6 + k, :r] = V_333[j*3 + k, :r]
        
        # W mapping:
        # W_333 has shape (9, r). Indices (k, i) -> k*3+i.
        # W_total has shape (18, rank_total). Indices (k, i) -> k*3+i.
        # For k in 0..2: W_total[k*3+i, d] = W_333[k*3+i, d]
        W_total[:9, :r] = W_333
        
        # Second half: columns r..2r-1
        U_total[:, r:] = U_333
        
        # V mapping for second half:
        # For k in 3..5: let k_local = k-3.
        # V_total[j*6+k, r+d'] = V_333[j*3+k_local, d']
        for j in range(3):
            for k_local in range(3):
                k = 3 + k_local
                V_total[j*6 + k, r:] = V_333[j*3 + k_local, :]
        
        # W mapping for second half:
        # For k in 3..5: let k_local = k-3.
        # W_total[k*3+i, r+d'] = W_333[k_local*3+i, d']
        for k_local in range(3):
            k = 3 + k_local
            W_total[k*3 : k*3+3, r:] = W_333[k_local*3 : k_local*3+3, :]
        
        # Verify
        T = _build_tensor()
        T_recon = np.einsum('ad,bd,cd->abc', U_total, V_total, W_total)
        if np.max(np.abs(T_recon - T)) < 1e-9:
            U_list = [[float(U_total[i, j]) for j in range(rank_total)] for i in range(n * m)]
            V_list = [[float(V_total[i, j]) for j in range(rank_total)] for i in range(m * p)]
            W_list = [[float(W_total[i, j]) for j in range(rank_total)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to original search
    T = _build_tensor()
    result = _search_rank(T, 54, n, m, p, time_limit)
    if result is not None:
        r, U, V, W = result
        U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
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
