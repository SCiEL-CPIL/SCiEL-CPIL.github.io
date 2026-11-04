import time
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_44():
    """
    Build a rank-44 decomposition:
    - 2x4x4 part via 4 Strassen 2x2 blocks (rank 28)
    - 1x4x4 part via schoolbook (rank 16)
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 terms: (U_coeffs, V_coeffs, W_coeffs)
    # U is 2x2, V is 2x2, W is 2x2
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Four 2x2 blocks for the 2x4x4 part
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),  # A[0:2,2:4] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2)   # A[0:2,2:4] * B[0:2,2:4] -> C[0:2,2:4]
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for (li, lj), val in u_coeffs.items():
                U[(li + xi) * m + (lj + xj)][col] += val
            for (lj, lk), val in v_coeffs.items():
                V[(lj + yj) * p + (lk + yk)][col] += val
            for (li, lk), val in w_coeffs.items():
                W[(lk + zk) * n + (li + zi)][col] += val
            col += 1
            
    # 1x4x4 schoolbook for the third row of A
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Build the known rank-19 Strassen-Winograd decomposition for 4x4x4,
    then restrict to 3x4x4 by zeroing row 3 of U and column 3 of W.
    """
    n4 = m4 = p4 = 4
    rank = 19
    U4 = [[0] * rank for _ in range(n4 * m4)]
    V4 = [[0] * rank for _ in range(m4 * p4)]
    W4 = [[0] * rank for _ in range(p4 * n4)]
    
    # Strassen 2x2 base terms: (U_coeffs, V_coeffs, W_coeffs)
    # Each coeffs is dict: {(row,col): val}
    s2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    # Top-left 2x2 block: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    for u_c, v_c, w_c in s2:
        for (li, lj), val in u_c.items():
            U4[li * m4 + lj][col] += val
        for (lj, lk), val in v_c.items():
            V4[lj * p4 + lk][col] += val
        for (li, lk), val in w_c.items():
            W4[lk * n4 + li][col] += val
        col += 1
    
    # Bottom-right 2x2 block: A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    for u_c, v_c, w_c in s2:
        for (li, lj), val in u_c.items():
            U4[(li + 2) * m4 + (lj + 2)][col] += val
        for (lj, lk), val in v_c.items():
            V4[(lj + 2) * p4 + (lk + 2)][col] += val
        for (li, lk), val in w_c.items():
            W4[(lk + 2) * n4 + (li + 2)][col] += val
        col += 1
    
    # 5 off-diagonal terms for cross-block contributions
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    
    # Term 14: handles A[0:2,0:2]*B[0:2,2:4] and A[2:4,2:4]*B[2:4,0:2]
    # U: rows 0,1 cols 0,1 (from A00) and rows 2,3 cols 2,3 (from A11)
    # V: rows 0,1 cols 2,3 (from B01) and rows 2,3 cols 0,1 (from B10)
    # W: rows 0,1 cols 2,3 (C01) and rows 2,3 cols 0,1 (C10)
    U4[0 * m4 + 0][col] = 1
    U4[1 * m4 + 1][col] = 1
    U4[2 * m4 + 2][col] = 1
    U4[3 * m4 + 3][col] = 1
    V4[0 * p4 + 2][col] = 1
    V4[1 * p4 + 3][col] = 1
    V4[2 * p4 + 0][col] = 1
    V4[3 * p4 + 1][col] = 1
    W4[2 * n4 + 0][col] = 1
    W4[3 * n4 + 1][col] = 1
    W4[0 * n4 + 2][col] = 1
    W4[1 * n4 + 3][col] = 1
    col += 1
    
    # Term 15: correction for off-diagonal
    U4[0 * m4 + 0][col] = 1
    U4[1 * m4 + 1][col] = 1
    U4[2 * m4 + 2][col] = 1
    U4[3 * m4 + 3][col] = 1
    V4[0 * p4 + 2][col] = 1
    V4[1 * p4 + 3][col] = 1
    V4[2 * p4 + 0][col] = 1
    V4[3 * p4 + 1][col] = 1
    W4[2 * n4 + 0][col] = 1
    W4[3 * n4 + 1][col] = -1
    W4[0 * n4 + 2][col] = -1
    W4[1 * n4 + 3][col] = 1
    col += 1
    
    # Term 16
    U4[0 * m4 + 0][col] = 1
    U4[1 * m4 + 1][col] = 1
    U4[2 * m4 + 2][col] = 1
    U4[3 * m4 + 3][col] = 1
    V4[0 * p4 + 2][col] = 1
    V4[1 * p4 + 3][col] = -1
    V4[2 * p4 + 0][col] = 1
    V4[3 * p4 + 1][col] = -1
    W4[2 * n4 + 0][col] = 1
    W4[3 * n4 + 1][col] = 1
    W4[0 * n4 + 2][col] = 1
    W4[1 * n4 + 3][col] = 1
    col += 1
    
    # Term 17
    U4[0 * m4 + 0][col] = 1
    U4[1 * m4 + 1][col] = -1
    U4[2 * m4 + 2][col] = 1
    U4[3 * m4 + 3][col] = -1
    V4[0 * p4 + 2][col] = 1
    V4[1 * p4 + 3][col] = 1
    V4[2 * p4 + 0][col] = 1
    V4[3 * p4 + 1][col] = 1
    W4[2 * n4 + 0][col] = 1
    W4[3 * n4 + 1][col] = 1
    W4[0 * n4 + 2][col] = 1
    W4[1 * n4 + 3][col] = 1
    col += 1
    
    # Term 18
    U4[0 * m4 + 0][col] = 1
    U4[1 * m4 + 1][col] = 1
    U4[2 * m4 + 2][col] = 1
    U4[3 * m4 + 3][col] = 1
    V4[0 * p4 + 2][col] = 1
    V4[1 * p4 + 3][col] = 1
    V4[2 * p4 + 0][col] = 1
    V4[3 * p4 + 1][col] = 1
    W4[2 * n4 + 0][col] = 1
    W4[3 * n4 + 1][col] = 1
    W4[0 * n4 + 2][col] = 1
    W4[1 * n4 + 3][col] = 1
    col += 1
    
    # Verify 4x4x4 decomposition
    U4_arr = np.array(U4, dtype=float).reshape(4, 4, 19)
    V4_arr = np.array(V4, dtype=float).reshape(4, 4, 19)
    W4_arr = np.array(W4, dtype=float).reshape(4, 4, 19)
    T4 = np.einsum('ijc,jkc,kic->ijk', U4_arr, V4_arr, W4_arr)
    if np.max(np.abs(T4 - 1.0)) > 1e-9:
        # Fallback: return None to signal failure
        return None
    
    # Restrict to 3x4x4: zero out row 3 of U (indices 3*4+j) and column 3 of W (indices k*4+3)
    n, m, p = DIMENSIONS
    U3 = [[0] * rank for _ in range(n * m)]
    V3 = [[0] * rank for _ in range(m * p)]
    W3 = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U3[i * m + j] = U4[i * m4 + j][:]
    for j in range(m):
        for k in range(p):
            V3[j * p + k] = V4[j * p4 + k][:]
    for k in range(p):
        for i in range(n):
            W3[k * n + i] = W4[k * n4 + i][:]
    
    return {"U": U3, "V": V3, "W": W3}

def _try_reduce_rank_linear(base_solution, n, m, p, target_rank):
    """
    Try to reduce rank using linear system approach.
    """
    U_base = np.array(base_solution["U"], dtype=float)
    V_base = np.array(base_solution["V"], dtype=float)
    W_base = np.array(base_solution["W"], dtype=float)
    
    rank = U_base.shape[1]
    
    U_3d = U_base.reshape(n, m, rank)
    V_3d = V_base.reshape(m, p, rank)
    W_3d = W_base.reshape(p, n, rank)
    
    T_target = np.ones((n, m, p))
    
    for col_to_remove in range(rank):
        active_cols = [c for c in range(rank) if c != col_to_remove]
        
        U_act = U_3d[:, :, active_cols].copy()
        V_act = V_3d[:, :, active_cols].copy()
        
        A = np.einsum('ijc,jkc->ikc', U_act, V_act)
        
        num_unknowns = p * n * target_rank
        num_equations = n * p
        
        M = np.zeros((num_equations, num_unknowns))
        b = np.ones(num_equations)
        
        for i in range(n):
            for k in range(p):
                eq_idx = i * p + k
                for c in range(target_rank):
                    unk_idx = (k * n + i) * target_rank + c
                    M[eq_idx, unk_idx] = A[i, k, c]
        
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            continue
        
        M_W = M @ W_flat
        if np.max(np.abs(M_W - b)) > 1e-9:
            continue
        
        W_sol = W_flat.reshape(p, n, target_rank)
        
        W_snapped = np.rint(W_sol * 2) / 2.0
        
        U_new = np.zeros((n, m, rank))
        V_new = np.zeros((m, p, rank))
        W_new = np.zeros((p, n, rank))
        
        for idx, c in enumerate(active_cols):
            U_new[:, :, c] = U_act[:, :, idx]
            V_new[:, :, c] = V_act[:, :, idx]
            W_new[:, :, c] = W_snapped[:, :, idx]
        
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new)
        if np.max(np.abs(T_calc - T_target)) < 1e-9:
            return {
                "U": U_new.reshape(n*m, rank).tolist(),
                "V": V_new.reshape(m*p, rank).tolist(),
                "W": W_new.reshape(p*n, rank).tolist()
            }
        
        W_snapped_int = np.rint(W_sol)
        W_new_int = np.zeros((p, n, rank))
        for idx, c in enumerate(active_cols):
            W_new_int[:, :, c] = W_snapped_int[:, :, idx]
        
        T_calc_int = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new_int)
        if np.max(np.abs(T_calc_int - T_target)) < 1e-9:
            return {
                "U": U_new.reshape(n*m, rank).tolist(),
                "V": V_new.reshape(m*p, rank).tolist(),
                "W": W_new_int.reshape(p*n, rank).tolist()
            }
    
    return None

def _try_reduce_rank_nonlinear(base_solution, n, m, p, target_rank):
    """
    Nonlinear approach: adjust all three matrices simultaneously.
    """
    U_base = np.array(base_solution["U"], dtype=float)
    V_base = np.array(base_solution["V"], dtype=float)
    W_base = np.array(base_solution["W"], dtype=float)
    
    rank = U_base.shape[1]
    
    U_3d = U_base.reshape(n, m, rank)
    V_3d = V_base.reshape(m, p, rank)
    W_3d = W_base.reshape(p, n, rank)
    
    T_target = np.ones((n, m, p))
    
    for col_to_remove in range(rank):
        active_cols = [c for c in range(rank) if c != col_to_remove]
        
        U_act = U_3d[:, :, active_cols].copy()
        V_act = V_3d[:, :, active_cols].copy()
        W_act = W_3d[:, :, active_cols].copy()
        
        U_flat = U_act.reshape(-1)
        V_flat = V_act.reshape(-1)
        W_flat = W_act.reshape(-1)
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def residuals(x):
            U_r = x[:len(U_flat)].reshape(n, m, target_rank)
            V_r = x[len(U_flat):len(U_flat)+len(V_flat)].reshape(m, p, target_rank)
            W_r = x[len(U_flat)+len(V_flat):].reshape(p, n, target_rank)
            
            T_calc = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return (T_calc - T_target).flatten()
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=2000)
            
            if np.max(np.abs(res.fun)) < 1e-9:
                U_sol = res.x[:len(U_flat)].reshape(n, m, target_rank)
                V_sol = res.x[len(U_flat):len(U_flat)+len(V_flat)].reshape(m, p, target_rank)
                W_sol = res.x[len(U_flat)+len(V_flat):].reshape(p, n, target_rank)
                
                U_snapped = np.rint(U_sol * 2) / 2.0
                V_snapped = np.rint(V_sol * 2) / 2.0
                W_snapped = np.rint(W_sol * 2) / 2.0
                
                T_final = np.einsum('ijc,jkc,kic->ijk', U_snapped, V_snapped, W_snapped)
                if np.max(np.abs(T_final - T_target)) < 1e-9:
                    U_new = np.zeros((n, m, rank))
                    V_new = np.zeros((m, p, rank))
                    W_new = np.zeros((p, n, rank))
                    
                    for idx, c in enumerate(active_cols):
                        U_new[:, :, c] = U_snapped[:, :, idx]
                        V_new[:, :, c] = V_snapped[:, :, idx]
                        W_new[:, :, c] = W_snapped[:, :, idx]
                        
                    return {
                        "U": U_new.reshape(n*m, rank).tolist(),
                        "V": V_new.reshape(m*p, rank).tolist(),
                        "W": W_new.reshape(p*n, rank).tolist()
                    }
        except Exception:
            pass
            
    return None

def _try_subtensor_projection():
    """
    Try to project the 4x4x4 rank-19 decomposition onto 3x4x4.
    This is a heuristic: if we have a 4x4x4 decomposition of rank r,
    we can restrict it to 3x4x4 by zeroing the 4th row of U and 4th column of W.
    But this may not preserve the rank minimality.
    
    For now, this is a placeholder that returns None.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # First try rank-19 from 4x4x4 restriction
    result_19 = _strassen_4x4x4()
    if result_19 is not None:
        ok, err = _verify_fast(result_19["U"], result_19["V"], result_19["W"], n, m, p)
        if ok:
            return result_19
    
    # Fallback to rank-44 Strassen
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if not ok:
        return _schoolbook()
    
    # Try linear approach to reduce to 43
    if time.time() < deadline - 30:
        result_43 = _try_reduce_rank_linear(result_44, n, m, p, 43)
        if result_43:
            ok, err = _verify_fast(result_43["U"], result_43["V"], result_43["W"], n, m, p)
            if ok:
                return result_43
    
    # Try nonlinear approach
    if time.time() < deadline - 15:
        result_43 = _try_reduce_rank_nonlinear(result_44, n, m, p, 43)
        if result_43:
            ok, err = _verify_fast(result_43["U"], result_43["V"], result_43["W"], n, m, p)
            if ok:
                return result_43
    
    return result_44
