import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    U46, V46, W46 = build_rank46()
    
    # Attempt to find Rank 45 using ALS + Snapping
    def try_rank_45():
        np.random.seed(123)
        rank = 45
        max_iters = 50
        for attempt in range(20):
            U = np.random.randn(U_dim, rank)
            V = np.random.randn(V_dim, rank)
            W = np.random.randn(W_dim, rank)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                # Update U
                # T reshaped: (U_dim, V_dim * W_dim)
                # T_u = T.reshape(U_dim, -1)
                # VW = V @ W.T  (V_dim x W_dim) -> (V_dim * W_dim) ? No.
                # We want to minimize || T - sum u_r v_r w_r^T ||
                # This is equivalent to minimizing || T_flat - U * (W * V.T) ||_F
                # Let's use standard ALS for CP decomposition.
                
                # Flatten T to (U_dim, V_dim * W_dim)
                T_uvw = T.reshape(U_dim, V_dim * W_dim)
                
                # Update U:
                # T_uvw approx U * (V @ W.T).T ?
                # T_{i,jk} = sum_r U_{i,r} V_{j,r} W_{k,r}
                # T_uvw[:, j*W_dim+k] = sum_r U[:,r] * (V[:,r] * W[:,r])
                # Let X = V * W (element-wise product, shape V_dim x rank)
                # Then T_uvw approx U * X.T ? No.
                # T_{i, j, k} = sum_r U_{i,r} (V_{j,r} W_{k,r})
                # Let Y = V * W (element-wise). Y is (V_dim, rank).
                # Then T_{i, j, k} = sum_r U_{i,r} Y_{j,r} W_{k,r} ... this is not linear in U alone if V,W fixed.
                # Actually, for fixed V, W:
                # T_{i, j, k} = sum_r U_{i,r} (V_{j,r} W_{k,r})
                # This can be viewed as:
                # T_flat[i, j*W_dim+k] = sum_r U_{i,r} * Z_{j*W_dim+k, r}
                # Where Z is constructed from V and W.
                # Z_{j, k, r} = V_{j,r} * W_{k,r}
                # Z_flat is (V_dim * W_dim, rank)
                
                Z = (V * W[None, :, :]).reshape(V_dim * W_dim, rank) # V_dim x W_dim x rank -> V_dim*W_dim x rank?
                # V is (V_dim, rank), W is (W_dim, rank)
                # V[:, None, :] * W[None, :, :] is (V_dim, W_dim, rank)
                # Reshape to (V_dim * W_dim, rank)
                
                # Solve U * Z = T_uvw
                # U = T_uvw @ Z.T @ (Z @ Z.T)^-1
                try:
                    ZZT = Z @ Z.T
                    U_new = T_uvw @ Z.T @ np.linalg.pinv(ZZT)
                except:
                    break
                
                # Update V
                T_vuw = T.transpose(1, 0, 2).reshape(V_dim, U_dim * W_dim)
                # T_{j, i, k} = sum_r V_{j,r} (U_{i,r} W_{k,r})
                Zv = (U * W[None, :, :]).reshape(U_dim * W_dim, rank)
                try:
                    ZZTv = Zv @ Zv.T
                    V_new = T_vuw @ Zv.T @ np.linalg.pinv(ZZTv)
                except:
                    break
                
                # Update W
                T_wuv = T.transpose(2, 0, 1).reshape(W_dim, U_dim * V_dim)
                # T_{k, i, j} = sum_r W_{k,r} (U_{i,r} V_{j,r})
                Zw = (U * V[None, :, :]).reshape(U_dim * V_dim, rank)
                try:
                    ZZTw = Zw @ Zw.T
                    W_new = T_wuv @ Zw.T @ np.linalg.pinv(ZZTw)
                except:
                    break
                    
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = np.linalg.norm(prod - T)
                if err < 1e-5:
                    U_s = snap_to_half(U)
                    V_s = snap_to_half(V)
                    W_s = snap_to_half(W)
                    if validate(U_s, V_s, W_s):
                        return U_s, V_s, W_s
            
            # Final check after loop
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            if validate(U_s, V_s, W_s):
                return U_s, V_s, W_s

        return None

    res_45 = try_rank_45()
    if res_45:
        U, V, W = res_45
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
