import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if indices are consistent
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try ranks from schoolbook down to 1
    max_rank = n * m * p
    # We can start from a slightly lower rank if we want, but 72 is safe.
    # Known lower bound is higher than 1.
    # Let's try a range.
    
    deadline = time.time() + 100  # 100 seconds to leave margin
    
    for R in range(max_rank, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] W[:, r])
                # This is not linear in U directly if we sum over r.
                # But we can vectorize.
                # T is dim_u x dim_v x dim_w.
                # We want U such that U @ (V^T @ W) ? No.
                
                # Let's reshape.
                # T_mat[a, b*w + c] = sum_r U[a, r] * (V[b, r] * W[c, r])
                # Let M_r = V[:, r] * W[:, r] (element-wise product)
                # Then T[a, b, c] = sum_r U[a, r] M_r[b, c]
                # This is a linear system in U[a, :].
                # T[a, b, c] = sum_r U[a, r] M_r[b, c]
                # Let X[a, r] = U[a, r].
                # Let Y[r, b, c] = M_r[b, c].
                # T[a, b, c] = sum_r X[a, r] Y[r, b, c]
                # This is equivalent to:
                # T_flat[a, b*c + c] = X[a, :] dot Y_flat[r, b*c + c]
                # So T_flat = X @ Y_flat^T
                # Where Y_flat is R x (dim_v * dim_w)
                
                # Construct Y_flat for current V, W
                # M_r[b, c] = V[b, r] * W[c, r]
                # Y_flat[r, b*dim_w + c] = V[b, r] * W[c, r]
                
                # Efficiently:
                # V is dim_v x R
                # W is dim_w x R
                # We need Y_flat of shape R x (dim_v * dim_w)
                # Y_flat[r, b*dim_w + c] = V[b, r] * W[c, r]
                
                # We can do this by broadcasting
                # V[:, :, None] * W[None, :, :] -> dim_v x R x dim_w
                # Reshape to R x dim_v x dim_w
                # Then flatten to R x (dim_v * dim_w)
                
                M = V[:, :, None] * W[None, :, :]  # dim_v x R x dim_w
                M = M.transpose(1, 0, 2)  # R x dim_v x dim_w
                Y_flat = M.reshape(R, dim_v * dim_w)
                
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                
                # Solve X @ Y_flat^T = T_flat
                # X = T_flat @ (Y_flat^T)^+
                # Use lstsq
                U_new, _, _, _ = np.linalg.lstsq(Y_flat.T, T_flat, rcond=None)
                U = U_new
                
                # Solve for V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, T[:, b, :] = sum_r (U[:, r] W[:, r]) V[b, r]
                # Let N_r[a, c] = U[a, r] * W[c, r]
                # T[a, b, c] = sum_r V[b, r] N_r[a, c]
                # T_flat_b = V[b, :] @ N_flat^T ?
                # T[a, b, c] = sum_r V[b, r] N_r[a, c]
                # Reshape T to dim_v x (dim_u * dim_w)
                # T_v[b, a*dim_w + c] = sum_r V[b, r] N_r[a, c]
                # N_flat[r, a*dim_w + c] = U[a, r] * W[c, r]
                
                N = U[:, :, None] * W[None, :, :]  # dim_u x R x dim_w
                N = N.transpose(1, 0, 2)  # R x dim_u x dim_w
                N_flat = N.reshape(R, dim_u * dim_w)
                
                T_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                
                V_new, _, _, _ = np.linalg.lstsq(N_flat.T, T_v, rcond=None)
                V = V_new
                
                # Solve for W
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed c, T[:, :, c] = sum_r (U[:, r] V[:, r]) W[c, r]
                # Let P_r[a, b] = U[a, r] * V[b, r]
                # T[a, b, c] = sum_r W[c, r] P_r[a, b]
                # T_w[c, a*dim_v + b] = sum_r W[c, r] P_r[a, b]
                # P_flat[r, a*dim_v + b] = U[a, r] * V[b, r]
                
                P = U[:, :, None] * V[None, :, :]  # dim_u x R x dim_v
                P = P.transpose(1, 0, 2)  # R x dim_u x dim_v
                P_flat = P.reshape(R, dim_u * dim_v)
                
                T_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                
                W_new, _, _, _ = np.linalg.lstsq(P_flat.T, T_w, rcond=None)
                W = W_new
                
            # Check residual
            # Reconstruct T_rec
            M = V[:, :, None] * W[None, :, :]
            M = M.transpose(1, 0, 2)
            Y_flat = M.reshape(R, dim_v * dim_w)
            T_rec_flat = U @ Y_flat.T
            T_rec = T_rec_flat.reshape(dim_u, dim_v, dim_w)
            
            err = np.linalg.norm(T - T_rec)
            if err < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                M_snap = V_snap[:, :, None] * W_snap[None, :, :]
                M_snap = M_snap.transpose(1, 0, 2)
                Y_flat_snap = M_snap.reshape(R, dim_v * dim_w)
                T_rec_snap_flat = U_snap @ Y_flat_snap.T
                T_rec_snap = T_rec_snap_flat.reshape(dim_u, dim_v, dim_w)
                
                if np.allclose(T_rec_snap, T, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
