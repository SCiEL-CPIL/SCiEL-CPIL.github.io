import time
import numpy as np
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses continuous optimization (Least Squares) to find low-rank decompositions,
    then snaps to half-integers.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    T_flat = T.flatten()
    
    # Fallback: Schoolbook decomposition (Rank 36)
    rank_schoolbook = n * m * p
    U_fb = np.zeros((dim_u, rank_schoolbook))
    V_fb = np.zeros((dim_v, rank_schoolbook))
    W_fb = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    def to_lists(mat):
        return mat.tolist()
    
    fallback_result = {"U": to_lists(U_fb), "V": to_lists(V_fb), "W": to_lists(W_fb)}
    
    def unpack(x, R):
        """Unpack flattened vector into U, V, W matrices."""
        len_u = dim_u * R
        len_v = dim_v * R
        len_w = dim_w * R
        
        U = x[:len_u].reshape(dim_u, R)
        V = x[len_u:len_u+len_v].reshape(dim_v, R)
        W = x[len_u+len_v:].reshape(dim_w, R)
        return U, V, W

    def residuals(x, R):
        """Calculate residuals for least_squares."""
        U, V, W = unpack(x, R)
        # Reconstruct tensor
        # T_recon = sum_r U[:,r] V[:,r] W[:,r]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (T_recon - T).flatten()

    def try_rank(R, num_restarts=5):
        """Attempt to find a decomposition of rank R."""
        if time.time() > deadline - 10:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
                
            # Initialize with random small values
            x0 = np.random.randn(dim_u * R + dim_v * R + dim_w * R) * 0.1
            
            try:
                res = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                # 'lm' (Levenberg-Marquardt) is good for least squares.
                # If it fails or takes too long, we might need to switch to 'trf' or 'dogbox'
            except Exception:
                # Fallback to BFGS if LM fails (e.g. due to constraints or step size)
                try:
                    # BFGS requires a scalar objective
                    def obj(x):
                        r = residuals(x, R)
                        return np.sum(r**2)
                    from scipy.optimize import minimize
                    res = minimize(obj, x0, method='BFGS', options={'maxiter': 100})
                except Exception:
                    continue

            final_res = res.fun
            final_error = np.sum(final_res**2)
            
            # Check if error is small enough to snap to half-integers
            # The tensor entries are 0 or 1.
            # If the continuous solution is close to the integer/half-integer lattice,
            # snapping should work.
            # Threshold: 1e-4 squared error per element?
            # Total elements: 6*18*12 = 1296.
            # If total error < 1e-6, avg error per element is very small.
            if final_error < 1e-6:
                U, V, W = unpack(res.x, R)
                
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                T_verify = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_verify, T, atol=1e-9):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
            # If error is not small enough, we can try to polish or accept if it's very close?
            # No, must be exact.
            
        return None

    # Try ranks in increasing order
    # Lower bound is 18.
    # Try 18, then 24, then 30, then 36 (fallback)
    # Given the structure, 18 might be hard. 24 is more likely if 18 is not possible.
    # But we want minimal.
    
    # Heuristic: Try 18 with more restarts if possible?
    # Let's try 18 first.
    result = try_rank(18, num_restarts=10)
    if result:
        return result
        
    if time.time() < deadline - 10:
        result = try_rank(24, num_restarts=10)
        if result:
            return result
            
    if time.time() < deadline - 5:
        result = try_rank(30, num_restarts=5)
        if result:
            return result
            
    return fallback_result
# EVOLVE-BLOCK-END
