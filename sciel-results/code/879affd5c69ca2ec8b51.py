import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor entries
    # T[i*m + j, j*p + k, k*n + i] = 1
    # All other entries are 0.
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Create a dense target tensor for fast comparison
    T_target = np.zeros((nm, mp, np_))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school

    def compute_tensor(U, V, W):
        """
        Computes the tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        U, V, W are shapes (nm, R), (mp, R), (np_, R)
        Returns T of shape (nm, mp, np_)
        """
        return np.einsum('ar,br,cr->abc', U, V, W)

    def residual_func(params, R):
        # params: flat array of U, V, W entries
        # Size: R * (nm + mp + np_)
        # U is (nm, R), V is (mp, R), W is (np_, R)
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm : R*nm + R*mp].reshape(mp, R)
        W = params[R*nm + R*mp : ].reshape(np_, R)
        
        T_calc = compute_tensor(U, V, W)
        
        # Residual is the difference from target
        return (T_calc - T_target).flatten()

    def is_valid_solution(U, V, W, R):
        # U, V, W are (nm, R), (mp, R), (np_, R)
        T_calc = compute_tensor(U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-6)

    def snap_to_half_int(matrix):
        # matrix is (rows, R)
        # Snap each entry to nearest 0.5
        return np.round(matrix * 2) / 2.0

    def check_snapped(U, V, W):
        T_calc = compute_tensor(U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-9)

    for R in range(1, best_rank + 1):
        if time.time() - start_time > timeout:
            break
            
        # Try more random starts due to faster residual calculation
        attempts = 50
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            params = np.random.randn(nm * R + mp * R + np_ * R)
            
            # Residual function for optimizer
            def res_func(p):
                return residual_func(p, R)
            
            try:
                result = least_squares(res_func, params, method='lm', max_nfev=500)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                # Extract solution
                U_sol = result.x[:R*nm].reshape(nm, R)
                V_sol = result.x[R*nm : R*nm + R*mp].reshape(mp, R)
                W_sol = result.x[R*nm + R*mp : ].reshape(np_, R)
                
                # Check if it's a good approximate solution
                if is_valid_solution(U_sol, V_sol, R):
                    # Try to snap to half-integers
                    U_snap = snap_to_half_int(U_sol)
                    V_snap = snap_to_half_int(V_sol)
                    W_snap = snap_to_half_int(W_sol)
                    
                    if check_snapped(U_snap, V_snap, W_snap):
                        # Found a valid half-integer solution
                        best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        return best_result

    return best_result
