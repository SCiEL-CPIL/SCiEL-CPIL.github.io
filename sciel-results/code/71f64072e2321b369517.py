import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    start_time = time.time()
    time_limit = 140 # Safety margin

    def is_valid(U, V, W, r):
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            prod_uv = u_c[:, :, None] * v_c[None, :, :]
            w_tensor = w_c.T[:, None, :]
            T += prod_uv * w_tensor
        return np.allclose(T, 1.0, atol=1e-6)

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    def try_reduce_rank(U, V, W, r):
        # Try removing each term
        for c_remove in range(r):
            if time.time() - start_time > time_limit:
                return None
            
            keep_idx = [i for i in range(r) if i != c_remove]
            
            U_keep = U[:, keep_idx]
            V_keep = V[:, keep_idx]
            W_keep = W[:, keep_idx]
            
            T_keep = np.zeros((n, m, p))
            for idx, c in enumerate(keep_idx):
                u_c = U_keep[:, idx].reshape(n, m)
                v_c = V_keep[:, idx].reshape(m, p)
                w_c = W_keep[:, idx].reshape(p, n)
                prod_uv = u_c[:, :, None] * v_c[None, :, :]
                w_tensor = w_c.T[:, None, :]
                T_keep += prod_uv * w_tensor
            
            b = 1.0 - T_keep
            b_flat = b.flatten()
            
            N_u = n * m
            N_v = m * p
            N_w = p * n
            N_vars = (N_u + N_v + N_w) * (r - 1)
            
            J = np.zeros((n * m * p, N_vars))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        
                        for idx, c in enumerate(keep_idx):
                            val_U = V_keep[j * p + k, idx] * W_keep[k * n + i, idx]
                            var_idx_U = i * m + j
                            J[eq_idx, var_idx_U * (r - 1) + idx] = val_U
                            
                            val_V = U_keep[i * m + j, idx] * W_keep[k * n + i, idx]
                            var_idx_V = j * p + k
                            J[eq_idx, (N_u * (r - 1)) + var_idx_V * (r - 1) + idx] = val_V
                            
                            val_W = U_keep[i * m + j, idx] * V_keep[j * p + k, idx]
                            var_idx_W = k * n + i
                            J[eq_idx, (N_u + N_v) * (r - 1) + var_idx_W * (r - 1) + idx] = val_W
            
            try:
                delta_p, residuals, rank_J, sv = np.linalg.lstsq(J, b_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            if np.linalg.norm(J.dot(delta_p) - b_flat) > 1e-5:
                continue
                
            U_svd, s, Vh = np.linalg.svd(J, full_matrices=False)
            
            found = False
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    return None
                    
                z = np.random.randn(N_vars)
                z_row = Vh.T.dot(Vh.dot(z))
                z_null = z - z_row
                
                delta = delta_p + z_null
                
                U_new = U_keep.copy()
                V_new = V_keep.copy()
                W_new = W_keep.copy()
                
                delta_U = delta[:N_u * (r - 1)].reshape(N_u, r - 1)
                delta_V = delta[N_u * (r - 1): (N_u + N_v) * (r - 1)].reshape(N_v, r - 1)
                delta_W = delta[(N_u + N_v) * (r - 1):].reshape(N_w, r - 1)
                
                U_new += delta_U
                V_new += delta_V
                W_new += delta_W
                
                U_snapped = snap_to_half_int(U_new)
                V_snapped = snap_to_half_int(V_new)
                W_snapped = snap_to_half_int(W_new)
                
                if is_valid(U_snapped, V_snapped, W_snapped, r - 1):
                    found = True
                    return U_snapped, V_snapped, W_snapped, r - 1
            
            if found:
                break
                
        return None

    # Main Loop
    while time.time() - start_time < time_limit:
        result = try_reduce_rank(best_U, best_V, best_W, best_rank)
        if result:
            best_U, best_V, best_W, best_rank = result
            continue
        else:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
