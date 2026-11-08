"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10)."""
import time
import numpy as np

def construct():
    _t0 = time.time()
    deadline = _t0 + 100  # 100s work, 50s safety
    
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T_target[a*m+b, b*p+d, d*n+a] = 1.0

    def verify(Um, Vm, Wm, atol=1e-9):
        T = np.einsum('il,jl,kl->ijk', Um, Vm, Wm)
        return np.allclose(T, T_target, atol=atol)

    # Verify base
    if not verify(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try eliminating each column
    for elim_col in range(rank):
        if time.time() > deadline:
            break
        
        mask = [c for c in range(rank) if c != elim_col]
        U_try = U_np[:, mask].copy()
        V_try = V_np[:, mask].copy()
        W_try = W_np[:, mask].copy()
        
        T_cur = np.einsum('il,jl,kl->ijk', U_try, V_try, W_try)
        residual = T_target - T_cur
        
        # Check if residual is rank-1 (tensor rank 1)
        # Reshape to matrix and check rank
        R_mat = residual.reshape(6, 600)
        r_rank = np.linalg.matrix_rank(R_mat, tol=1e-8)
        if r_rank > 1:
            continue
        
        # Fit rank-1 tensor to residual
        u = np.zeros(6)
        v = np.zeros(30)
        w = np.zeros(20)
        
        # SVD initialization
        U_svd, s_vals, Vt = np.linalg.svd(R_mat, full_matrices=False)
        if len(s_vals) > 0 and s_vals[0] > 1e-12:
            u = U_svd[:, 0].copy()
            vw_mat = s_vals[0] * Vt[0].reshape(30, 20)
            V_svd, s2, Wt = np.linalg.svd(vw_mat, full_matrices=False)
            if len(s2) > 0 and s2[0] > 1e-12:
                v = V_svd[:, 0].copy()
                w = s2[0] * Wt[0].copy()
        
        # Alternating least squares
        for it in range(100):
            if time.time() > deadline:
                break
            # Update u
            vw = np.outer(v, w)
            u_new = np.array([np.sum(residual[i] * vw) for i in range(6)])
            denom = np.sum(vw**2)
            if abs(denom) > 1e-15:
                u = u_new / denom
            
            # Update v
            uw = np.outer(u, w)
            v_new = np.array([np.sum(residual[:, j, :] * uw) for j in range(30)])
            denom = np.sum(uw**2)
            if abs(denom) > 1e-15:
                v = v_new / denom
            
            # Update w
            uv = np.outer(u, v)
            w_new = np.array([np.sum(residual[:, :, k] * uv) for k in range(20)])
            denom = np.sum(uv**2)
            if abs(denom) > 1e-15:
                w = w_new / denom
        
        # Check solution quality
        T_new = T_cur + np.einsum('i,j,k->ijk', u, v, w)
        if np.allclose(T_new, T_target, atol=1e-8):
            U_new = np.column_stack([U_try, u])
            V_new = np.column_stack([V_try, v])
            W_new = np.column_stack([W_try, w])
            
            new_rank = U_new.shape[1]
            if new_rank < best_rank:
                best_rank = new_rank
                # Convert to lists
                best_U = [[float(x) for x in U_new[i]] for i in range(6)]
                best_V = [[float(x) for x in V_new[i]] for i in range(30)]
                best_W = [[float(x) for x in W_new[i]] for i in range(20)]
                # Update working copies for further elimination attempts
                U_np = U_new.copy()
                V_np = V_new.copy()
                W_np = W_new.copy()
                rank = new_rank
                break  # Restart elimination loop

    if time.time() > deadline - 3:
        return {"U": best_U, "V": best_V, "W": best_W}
    return {"U": best_U, "V": best_V, "W": best_W}
