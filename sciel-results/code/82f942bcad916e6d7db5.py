"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # The tensor T is sparse.
    # We want to find U, V, W of rank R such that U @ V @ W approximates the tensor.
    # Actually, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # This is equivalent to T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    
    # We can flatten the tensor T into a vector t of length nm * mp * pn.
    # But it's sparse.
    # Let's use the 84 non-zero entries.
    
    # Indices of non-zero entries
    indices_i = []
    indices_j = []
    indices_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_i.append(i * m + j)
                indices_j.append(j * p + k)
                indices_k.append(k * n + i)
                
    indices_i = np.array(indices_i)
    indices_j = np.array(indices_j)
    indices_k = np.array(indices_k)
    
    def error_func(params, R):
        # params is a flattened array of U, V, W
        # U: nm x R, V: mp x R, W: pn x R
        nm, mp, pn = n * m, m * p, p * n
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape((nm, R))
        V = V_flat.reshape((mp, R))
        W = W_flat.reshape((pn, R))
        
        # Calculate T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We only care about the 84 non-zero entries.
        # T_nonzero = sum_r U[indices_i, r] V[indices_j, r] W[indices_k, r]
        
        # Efficient calculation:
        # U_sel = U[indices_i, :] (84 x R)
        # V_sel = V[indices_j, :] (84 x R)
        # W_sel = W[indices_k, :] (84 x R)
        # T_nonzero = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        U_sel = U[indices_i, :]
        V_sel = V[indices_j, :]
        W_sel = W[indices_k, :]
        
        T_nonzero = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        # The target is 1 for all these entries.
        return T_nonzero - 1.0

    def try_rank(R, max_restarts=5, max_iters=2000):
        nm, mp, pn = n * m, m * p, p * n
        total_params = R * (nm + mp + pn)
        
        for restart in range(max_restarts):
            # Initialize with small random values
            params0 = np.random.randn(total_params) * 0.1
            # Or initialize with schoolbook solution if R >= schoolbook rank?
            # If R < schoolbook, we can't just copy.
            
            try:
                # Use scipy.optimize.least_squares
                # We need to import scipy
                from scipy.optimize import least_squares
                result = least_squares(error_func, params0, args=(R,), method='lm', max_nfev=max_iters)
                if result.success or np.max(np.abs(result.fun)) < 1e-9:
                    # Check if we can snap to integers/half-integers
                    # This is a heuristic.
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:nm*R+mp*R+pn*R]
                    
                    U = U_flat.reshape((nm, R))
                    V = V_flat.reshape((mp, R))
                    W = W_flat.reshape((pn, R))
                    
                    # Snap to nearest integer or half-integer
                    # Multiply by 2, round, divide by 2
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify the snapped solution
                    U_sel = U_snap[indices_i, :]
                    V_sel = V_snap[indices_j, :]
                    W_sel = W_snap[indices_k, :]
                    T_snap = np.sum(U_sel * V_sel * W_sel, axis=1)
                    
                    if np.max(np.abs(T_snap - 1.0)) < 1e-9:
                        return U_snap, V_snap, W_snap
            except Exception:
                pass
        return None

    # Try to find a lower rank
    # Start from 83 down to 70
    # This might be slow, so we limit the time.
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline:
            break
        res = try_rank(R, max_restarts=2, max_iters=500)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no improvement found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
