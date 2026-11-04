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
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
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
            
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _build_linear_system(n, m, p, target_rank):
    """
    Build the linear system for the tensor identity.
    
    The tensor identity is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    This is nonlinear in the variables. However, we can linearize by fixing
    two of the three matrices and solving for the third.
    
    For a given column c, the contribution to T[i,j,k] is:
    U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    If we fix U and V for all columns except one, and want to solve for W,
    the system becomes linear in W.
    
    Strategy: Fix U and V from the base solution (with one column removed),
    then solve for W. But this may not work because the system is still
    underdetermined.
    
    Better strategy: Use the "Lax subspace" approach.
    For each output position (i,k), the equation is:
    sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    This can be rewritten as:
    sum_c W[k,i,c] * (sum_j U[i,j,c] * V[j,k,c]) = 1
    
    Let A[i,k,c] = sum_j U[i,j,c] * V[j,k,c]
    Then: sum_c A[i,k,c] * W[k,i,c] = 1
    
    This is linear in W! We can solve for W given U and V.
    """
    n, m, p = n, m, p
    # A[i,k,c] = sum_j U[i,j,c] * V[j,k,c]
    # Shape: (n, p, target_rank)
    # W[k,i,c] has shape (p, n, target_rank)
    # Equation: sum_c A[i,k,c] * W[k,i,c] = 1 for each (i,k)
    
    # We'll build the system: for each (i,k), sum_c A[i,k,c] * W[k,i,c] = 1
    # Unknowns: W[k,i,c] for all k,i,c -> p*n*target_rank unknowns
    # Equations: n*p equations
    
    return n, m, p, target_rank

def _try_reduce_rank_linear(base_solution, n, m, p, target_rank):
    """
    Try to reduce rank using linear system approach.
    
    For each column to remove:
    1. Take the base solution, remove one column
    2. Fix U and V (with the removed column zeroed)
    3. Solve the linear system for W
    4. Check if W has integer/half-integer values
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
        # Create active columns
        active_cols = [c for c in range(rank) if c != col_to_remove]
        
        # Extract active parts
        U_act = U_3d[:, :, active_cols].copy()  # (n, m, target_rank)
        V_act = V_3d[:, :, active_cols].copy()  # (m, p, target_rank)
        
        # Build A[i,k,c] = sum_j U[i,j,c] * V[j,k,c]
        # A shape: (n, p, target_rank)
        A = np.einsum('ijc,jkc->ikc', U_act, V_act)
        
        # Now solve: sum_c A[i,k,c] * W[k,i,c] = 1 for all (i,k)
        # Unknowns: W[k,i,c] for k in range(p), i in range(n), c in range(target_rank)
        # Total unknowns: p * n * target_rank
        # Total equations: n * p
        
        num_unknowns = p * n * target_rank
        num_equations = n * p
        
        # Build the system matrix
        # For equation (i,k): sum_c A[i,k,c] * W[k,i,c] = 1
        # W[k,i,c] is at position (k * n + i) * target_rank + c in the unknown vector
        
        # We'll use least squares since the system is underdetermined
        # But first, let's try to find a particular solution
        
        # Build the matrix M where M[equation_idx, unknown_idx] = coefficient
        # equation_idx = i * p + k
        # unknown_idx = (k * n + i) * target_rank + c
        
        M = np.zeros((num_equations, num_unknowns))
        b = np.ones(num_equations)
        
        for i in range(n):
            for k in range(p):
                eq_idx = i * p + k
                for c in range(target_rank):
                    unk_idx = (k * n + i) * target_rank + c
                    M[eq_idx, unk_idx] = A[i, k, c]
        
        # Solve the linear system using least squares
        # Since it's underdetermined, we want the minimum-norm solution
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            continue
        
        # Check if the solution satisfies the equations
        M_W = M @ W_flat
        if np.max(np.abs(M_W - b)) > 1e-9:
            continue
        
        # Reshape W
        W_sol = W_flat.reshape(p, n, target_rank)
        
        # Try to snap to integers or half-integers
        W_snapped = np.rint(W_sol * 2) / 2.0
        
        # Verify the full decomposition
        U_new = np.zeros((n, m, rank))
        V_new = np.zeros((m, p, rank))
        W_new = np.zeros((p, n, rank))
        
        for idx, c in enumerate(active_cols):
            U_new[:, :, c] = U_act[:, :, idx]
            V_new[:, :, c] = V_act[:, :, idx]
            W_new[:, :, c] = W_snapped[:, :, idx]
        
        # Verify
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new)
        if np.max(np.abs(T_calc - T_target)) < 1e-9:
            return {
                "U": U_new.reshape(n*m, rank).tolist(),
                "V": V_new.reshape(m*p, rank).tolist(),
                "W": W_new.reshape(p*n, rank).tolist()
            }
        
        # Also try integer snapping
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
        
        # Try the original (non-snapped) W to see if it works
        W_new_orig = np.zeros((p, n, rank))
        for idx, c in enumerate(active_cols):
            W_new_orig[:, :, c] = W_sol[:, :, idx]
        
        T_calc_orig = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new_orig)
        if np.max(np.abs(T_calc_orig - T_target)) < 1e-9:
            # Check if values are close to integers or half-integers
            all_vals = np.concatenate([U_new.flatten(), V_new.flatten(), W_new_orig.flatten()])
            snapped = np.rint(all_vals * 2) / 2.0
            if np.max(np.abs(all_vals - snapped)) < 1e-9:
                return {
                    "U": U_new.reshape(n*m, rank).tolist(),
                    "V": V_new.reshape(m*p, rank).tolist(),
                    "W": W_new_orig.reshape(p*n, rank).tolist()
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
                
                # Snap to half-integers
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
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
