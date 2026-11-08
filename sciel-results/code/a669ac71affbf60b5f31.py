import time
import numpy as np

_DIM = (6, 6, 6)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _residual_max(U, V, W, n, m, p):
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # Use einsum for speed
    R_3d = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = np.max(np.abs(R_3d - 1.0))
    return float(residual)

def construct():
    n, m, p = _DIM
    deadline = time.time() + 115.0  # Safety margin

    U, V, W = _schoolbook(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_res = _residual_max(U, V, W, n, m, p)
    best_rank = U.shape[1]

    rng = np.random.default_rng(12345)
    
    def try_drop(U, V, W, rank, drop_frac):
        if time.time() > deadline:
            return None
            
        cols = np.arange(rank)
        perm = rng.permutation(cols)
        n_drop = int(round(rank * drop_frac))
        drop_list = perm[:n_drop].tolist()
        keep_list = perm[n_drop:].tolist()
        
        if len(keep_list) < max(1, rank // 4):
            return None
            
        drop_idx = np.array(drop_list, dtype=np.int64)
        keep_idx = np.array(keep_list, dtype=np.int64)
        
        U0 = U.copy(); V0 = V.copy(); W0 = W.copy()
        U0[:, drop_idx] = 0.0
        V0[:, drop_idx] = 0.0
        W0[:, drop_idx] = 0.0
        Uk = U0[:, keep_idx]
        Vk = V0[:, keep_idx]
        Wk = W0[:, keep_idx]
        
        nm, mp, pn = n * m, m * p, p * n
        R = len(keep_idx)
        
        # C_flat shape: (n * p * m,)
        C_flat = np.ones(n * p * m)
        
        # Construct M vectorized
        # M_U, M_V, M_W shapes: (n*p*m, R * dim)
        
        # Precompute index arrays for rows
        # Row index (i, k, j) -> i*p*m + k*m + j
        i_idx, k_idx, j_idx = np.indices((n, p, m))
        row_flat = i_idx.ravel() * p * m + k_idx.ravel() * m + j_idx.ravel()
        
        # Reshape U, V, W to (R, n, m) etc
        Uk3 = Uk.T.reshape(R, n, m) # (R, n, m)
        Vk3 = Vk.T.reshape(R, m, p) # (R, m, p)
        Wk3 = Wk.T.reshape(R, p, n) # (R, p, n)
        
        M_U = np.zeros((n * p * m, R * nm))
        M_V = np.zeros((n * p * m, R * mp))
        M_W = np.zeros((n * p * m, R * pn))
        
        # Vectorized construction of M blocks
        # For M_U:
        # M_U[row, r*nm + i*m + j] = Vk3[r, j, k] * Wk3[r, k, i]
        # We can use einsum to build the full tensor then reshape?
        # Tensor shape for coeff: (R, n, p, m, n, m) ? No.
        
        # Let's stick to the loop but make it faster by precomputing row indices for each (i,j), (j,k), (k,i)
        # But the previous code did that. The error was likely in the indexing logic or data types.
        # Let's try to build M using matrix multiplications of permutation matrices?
        
        # Alternative: Use the fact that M is linear in U, V, W.
        # We can just solve the system using the current U, V, W as the "basis" and find the projection?
        # No, we are dropping columns and re-solving for the remaining ones to minimize error.
        
        # Let's try a simpler vectorized approach for M_U.
        # Coeff_U[i, k, j, r, i2, j2] = Vk3[r, j2, k] * Wk3[r, k, i2]
        # We want to place this into M_U.
        
        # This is too complex to vectorize safely without significant memory overhead (R * 216 * 36 ~ 1.5M floats, which is fine).
        
        # Let's try to build M_U block by block (for each r) but using advanced indexing correctly.
        
        try:
            for r in range(R):
                uk = Uk3[r] # (n, m)
                vk = Vk3[r] # (m, p)
                wk = Wk3[r] # (p, n)
                
                # M_U block for r
                # coeff[i, k, j] = wk[k, i] * vk[j, k]
                # wk.T is (n, p). vk.T is (p, m).
                # coeff[i, k, j] = wk.T[i, k] * vk.T[k, j]
                wk_T = wk.T
                vk_T = vk.T
                # coeff shape (n, p, m)
                coeff_U = wk_T[:, :, np.newaxis] * vk_T[np.newaxis, :, :]
                
                # Flatten coeff_U to (n*p*m,)
                coeff_U_flat = coeff_U.reshape(-1)
                
                # We need to scatter this into M_U columns r*nm + i*m + j
                # For each (i, j), the values are at rows where row_i==i and row_j==j.
                # Rows are (i, k, j).
                # So for fixed i, j, the rows are i*p*m + k*m + j for k in 0..p-1.
                
                # We can precompute these row indices for all (i, j)
                # rows_for_ij[i, j] = [i*p*m + k*m + j for k in range(p)]
                
                # Let's create a 2D array of row indices for U
                # u_rows[i, j, k] = i*p*m + k*m + j
                u_rows = np.zeros((n, m, p), dtype=np.int64)
                for i in range(n):
                    for j in range(m):
                        u_rows[i, j, :] = i * p * m + np.arange(p) * m + j
                
                # Now, for each r, we want to assign coeff_U_flat to M_U
                # M_U[rows, col] = val
                # This is equivalent to:
                # For each (i, j), M_U[u_rows[i, j, :], r*nm + i*m + j] = coeff_U[i, :, j]
                
                # We can vectorize this assignment using advanced indexing.
                # Flatten u_rows to (n*m, p)
                u_rows_flat = u_rows.reshape(n * m, p)
                # Flatten coeff_U to (n*m, p) ?
                # coeff_U is (n, p, m). We want (n, m, p) to match u_rows.
                coeff_U_rearranged = np.transpose(coeff_U, (0, 2, 1)) # (n, m, p)
                coeff_U_flat_2d = coeff_U_rearranged.reshape(n * m, p)
                
                # Column indices for this r
                # cols_u_r = r*nm + i*m + j
                # We need to map (i, j) to col index.
                # col_idx_u = r * nm + np.arange(n * m)
                
                col_idx_u = r * nm + np.arange(n * m)
                
                # Now we have:
                # Rows: u_rows_flat (n*m, p)
                # Cols: col_idx_u (n*m,)
                # Vals: coeff_U_flat_2d (n*m, p)
                
                # We need to assign M_U[rows[i, k], cols[i]] = vals[i, k] for i in 0..nm-1, k in 0..p-1
                # This is a scatter operation.
                # M_U is (n*p*m, R*nm).
                
                # We can use a loop over the n*m blocks (36 iterations) which is fast.
                for idx in range(n * m):
                    rows = u_rows_flat[idx]
                    col = col_idx_u[idx]
                    vals = coeff_U_flat_2d[idx]
                    M_U[rows, col] = vals
                
                # Similarly for M_V
                # coeff_V[i, k, j] = wk[k, i] * uk[i, j]
                # wk is (p, n). uk is (n, m).
                # coeff_V[k, i, j] = wk[k, i] * uk[i, j]
                coeff_V_raw = wk[:, :, np.newaxis] * uk[np.newaxis, :, :] # (p, n, m)
                coeff_V = np.transpose(coeff_V_raw, (1, 0, 2)) # (n, p, m)
                
                # Rows for V: (i, k, j) -> i*p*m + k*m + j
                # For fixed (j, k), rows are i*p*m + k*m + j for i in 0..n-1.
                v_rows = np.zeros((m, p, n), dtype=np.int64)
                for j in range(m):
                    for k in range(p):
                        v_rows[j, k, :] = np.arange(n) * p * m + k * m + j
                        
                v_rows_flat = v_rows.reshape(m * p, n)
                coeff_V_rearranged = np.transpose(coeff_V, (2, 1, 0)) # (m, p, n) ?
                # coeff_V is (n, p, m). We want (j, k, i) order?
                # v_rows is indexed by (j, k, i).
                # coeff_V[i, k, j]. We want coeff_V[j, k, i] to match v_rows[j, k, i]?
                # No, v_rows[j, k, i] is the row index for i.
                # The value is coeff_V[i, k, j].
                # So for fixed j, k, the values for i are coeff_V[:, k, j].
                # coeff_V is (n, p, m). coeff_V[:, k, j] is (n,).
                # So we need to extract columns from coeff_V.
                
                col_idx_v = r * mp + np.arange(m * p)
                
                for idx in range(m * p):
                    j = idx // p
                    k = idx % p
                    rows = v_rows_flat[idx]
                    col = col_idx_v[idx]
                    vals = coeff_V[:, k, j] # (n,)
                    M_V[rows, col] = vals
                
                # Similarly for M_W
                # coeff_W[i, k, j] = uk[i, j] * vk[j, k]
                # uk (n, m), vk (m, p).
                # coeff_W[i, j, k] = uk[i, j] * vk[j, k]
                coeff_W_raw = uk[:, :, np.newaxis] * vk[np.newaxis, :, :] # (n, m, p)
                coeff_W = np.transpose(coeff_W_raw, (0, 2, 1)) # (n, p, m)
                
                # Rows for W: (i, k, j) -> i*p*m + k*m + j
                # For fixed (k, i), rows are i*p*m + k*m + j for j in 0..m-1.
                w_rows = np.zeros((p, n, m), dtype=np.int64)
                for k in range(p):
                    for i in range(n):
                        w_rows[k, i, :] = i * p * m + k * m + np.arange(m)
                        
                w_rows_flat = w_rows.reshape(p * n, m)
                
                col_idx_w = r * pn + np.arange(p * n)
                
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    rows = w_rows_flat[idx]
                    col = col_idx_w[idx]
                    vals = coeff_W[i, k, :] # (m,)
                    M_W[rows, col] = vals
            
            M = np.hstack([M_U, M_V, M_W])
            
            x, *_ = np.linalg.lstsq(M, C_flat, rcond=None)
            
            Uk_new = x[0: nm * R].reshape(nm, R)
            Vk_new = x[nm * R: nm * R + mp * R].reshape(mp, R)
            Wk_new = x[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
            
            U_new = U0.copy(); V_new = V0.copy(); W_new = W0.copy()
            U_new[:, keep_idx] = Uk_new
            V_new[:, keep_idx] = Vk_new
            W_new[:, keep_idx] = Wk_new
            
            res = _residual_max(U_new, V_new, W_new, n, m, p)
            if res < 1e-7:
                return (U_new, V_new, W_new, res, len(keep_idx))
            return None
        except Exception:
            return None

    cur_U, cur_V, cur_W = U, V, W
    cur_rank = cur_U.shape[1]
    drop_frac = 0.05
    stall = 0
    
    while time.time() < deadline:
        if stall > 10: # Reduced stall limit as vectorized try_drop is faster
            break
            
        result = try_drop(cur_U, cur_V, cur_W, cur_rank, drop_frac)
        if result is not None:
            cur_U, cur_V, cur_W, res, new_rank = result
            cur_rank = new_rank
            stall = 0
            if new_rank < best_rank:
                best = (cur_U.copy(), cur_V.copy(), cur_W.copy())
                best_rank = new_rank
                best_res = res
            drop_frac = min(0.25, drop_frac * 1.1)
        else:
            stall += 1
            drop_frac = max(0.02, drop_frac * 0.9)

    Uf, Vf, Wf = best
    Uo = [[round(float(v), 9) for v in row] for row in Uf]
    Vo = [[round(float(v), 9) for v in row] for row in Vf]
    Wo = [[round(float(v), 9) for v in row] for row in Wf]
    return {"U": Uo, "V": Vo, "W": Wo}
