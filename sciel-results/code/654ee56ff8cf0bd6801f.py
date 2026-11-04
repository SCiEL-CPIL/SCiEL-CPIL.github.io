import numpy as np
import time
import itertools

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _block_specs():
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return False
    return True

def _build_tensor_matrix(U, V, W, n, m, p):
    """Build the tensor matrix M where M[idx, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    R = U.shape[1]
    dim = n * p * m
    M = np.zeros((dim, R))
    for r in range(R):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx = i * p * m + k * m + j
                    M[idx, r] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    return M

def _greedy_select_columns(M, target_rank, deadline):
    """Greedy forward selection with efficient residual updates."""
    dim, R = M.shape
    if target_rank >= R:
        return list(range(R))
    
    # Compute column norms
    col_norms = np.linalg.norm(M, axis=0)
    
    # Start with the column with largest norm
    selected = [int(np.argmax(col_norms))]
    remaining = set(range(R)) - {selected[0]}
    
    # Current projection: P = M_sel @ (M_sel^T M_sel)^{-1} M_sel^T
    # We maintain the residual R_res = M - P M for remaining columns
    # Actually, we track the residual for each remaining column
    
    # Compute residual for each remaining column
    M_sel = M[:, selected]
    # Solve for coefficients of each remaining column in span of selected
    # residual_j = M[:, j] - M_sel @ (M_sel^T M_sel)^{-1} M_sel^T M[:, j]
    
    # Use the fact that we can compute all residuals at once
    # Let G = M_sel^T M_sel (k x k), and H = M_sel^T M (k x dim)
    # Then residual for column j = M[:, j] - M_sel @ G^{-1} H[:, j]
    # residual_norm^2 = ||M[:, j]||^2 - H[:, j]^T G^{-1} H[:, j]
    
    G = M_sel.T @ M_sel
    H = M_sel.T @ M
    
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinalgError:
        G_inv = np.linalg.pinv(G)
    
    # Compute residual norms for all remaining columns
    # res_norm_sq[j] = col_norms[j]^2 - H[:, j]^T G_inv H[:, j]
    H_rem = H[:, list(remaining)]
    res_norm_sq = col_norms[list(remaining)]**2 - np.sum(H_rem.T @ G_inv @ H_rem, axis=0)
    res_norm_sq = np.maximum(res_norm_sq, 0)  # numerical safety
    res_norms = np.sqrt(res_norm_sq)
    
    rem_list = sorted(remaining)
    res_norms_arr = res_norms
    
    for step in range(1, target_rank):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Find column with max residual norm
        best_local = int(np.argmax(res_norms_arr))
        best_col = rem_list[best_local]
        best_norm = res_norms_arr[best_local]
        
        if best_norm < 1e-8:
            break
        
        selected.append(best_col)
        remaining.discard(best_col)
        rem_list.remove(best_col)
        res_norms_arr = np.delete(res_norms_arr, best_local)
        
        # Update G and H with new column
        M_new_col = M[:, best_col]
        # New G: [[G, g], [g^T, g_new]] where g = M_sel^T M_new_col, g_new = ||M_new_col||^2
        g = M_sel.T @ M_new_col
        g_new = col_norms[best_col]**2
        
        G_new = np.zeros((len(selected), len(selected)))
        G_new[:-1, :-1] = G
        G_new[:-1, -1] = g
        G_new[-1, :-1] = g
        G_new[-1, -1] = g_new
        
        # Update H: new row is M_sel^T M for all remaining
        H_new_row = M[:, list(remaining)].T @ M_new_col  # this is wrong dimension
        # Actually H is (k x dim), we need to add a new row: M_new_col^T M_remaining
        H_new_row = M_new_col @ M[:, list(remaining)].T  # (1 x |remaining|)
        
        H_new = np.zeros((len(selected), len(remaining)))
        H_new[:-1, :] = H[:, list(remaining)]
        H_new[-1, :] = H_new_row
        
        G = G_new
        H = H_new
        
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        
        # Update residual norms
        H_rem = H[:, list(remaining)]
        res_norm_sq = col_norms[list(remaining)]**2 - np.sum(H_rem.T @ G_inv @ H_rem, axis=0)
        res_norm_sq = np.maximum(res_norm_sq, 0)
        res_norms_arr = np.sqrt(res_norm_sq)
    
    return selected

def _try_column_elimination(U, V, W, n, m, p, deadline):
    """Try removing each column and check if the remaining still span the tensor."""
    R = U.shape[1]
    if R <= 1:
        return U, V, W
    
    # Build the tensor matrix
    M = _build_tensor_matrix(U, V, W, n, m, p)
    dim = M.shape[0]
    
    # Find the rank of M
    s_vals = np.linalg.svd(M, compute_uv=False)
    tol = 1e-10 * s_vals[0] * max(M.shape)
    target_rank = int(np.sum(s_vals > tol))
    
    if target_rank >= R:
        return U, V, W
    
    # Try greedy selection
    selected = _greedy_select_columns(M, target_rank, deadline)
    
    if len(selected) < R:
        keep = sorted(selected)
        U_try = U[:, keep]
        V_try = V[:, keep]
        W_try = W[:, keep]
        if _verify(U_try, V_try, W_try, n, m, p):
            return U_try, V_try, W_try
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0

    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"

    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]

    U_final = U
    V_final = V
    W_final = W

    try:
        # First pass: greedy column selection
        if time.time() < deadline - 20.0:
            U_final, V_final, W_final = _try_column_elimination(U, V, W, n, m, p, deadline - 10.0)
    except Exception:
        pass

    # Second pass: try to further reduce if we got a reduction
    try:
        if time.time() < deadline - 10.0:
            R_cur = U_final.shape[1]
            if R_cur < R:
                U_final, V_final, W_final = _try_column_elimination(U_final, V_final, W_final, n, m, p, deadline - 5.0)
    except Exception:
        pass

    # Final verification
    if time.time() < deadline - 2.0:
        if not _verify(U_final, V_final, W_final, n, m, p):
            U_final = U
            V_final = V
            W_final = W

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
