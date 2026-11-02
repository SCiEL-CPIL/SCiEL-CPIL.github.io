import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def compute_residual(U, V, W):
        # U, V, W are flat arrays of length R * nm, etc.
        R = U.shape[0] // nm
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # einsum: 'ijr,jkr,kin->ijk'
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        return T - 1.0

    def residual(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        res = compute_residual(U_flat, V_flat, W_flat)
        return res.flatten()

    def verify_half_integer(U, V, W):
        """Check if U, V, W are exactly half-integers and decompose correctly."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2)):
            return False
        if not np.allclose(V * 2, np.round(V * 2)):
            return False
        if not np.allclose(W * 2, np.round(W * 2)):
            return False
        
        # Verify decomposition
        res = compute_residual(U, V, W)
        return np.allclose(res, 0.0, atol=1e-10)

    def snap_to_half_integer(arr):
        """Snap array to nearest half-integers."""
        return np.round(arr * 2) / 2

    def local_search(U, V, W, R, max_iters=50):
        """
        Perform local search on the half-integer lattice to minimize residual.
        Returns (U, V, W, success)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_res = np.sum(compute_residual(U, V, W)**2)
        
        for _ in range(max_iters):
            improved = False
            # Try flipping each variable by +/- 0.5
            # We iterate over all variables in U, V, W
            # To be efficient, we can just recalculate residual for each candidate
            # But that's O(N * Cost). N = R*(nm+mp+pn). Cost = O(nmp R).
            # For R=20, N=620, Cost=600. 620*600 = 372,000 ops per sweep.
            # 50 sweeps = 18M ops. Fast enough.
            
            # Flatten all variables
            all_vars = np.concatenate([U, V, W])
            n_vars = len(all_vars)
            
            for idx in range(n_vars):
                # Determine which array and index
                if idx < nm * R:
                    arr = U
                    arr_idx = idx
                elif idx < nm * R + mp * R:
                    arr = V
                    arr_idx = idx - nm * R
                else:
                    arr = W
                    arr_idx = idx - nm * R - mp * R
                    
                orig_val = arr[arr_idx]
                
                # Try +0.5
                arr[arr_idx] = orig_val + 0.5
                new_res = np.sum(compute_residual(U, V, W)**2)
                if new_res < current_res - 1e-12:
                    current_res = new_res
                    improved = True
                else:
                    # Try -0.5
                    arr[arr_idx] = orig_val - 0.5
                    new_res = np.sum(compute_residual(U, V, W)**2)
                    if new_res < current_res - 1e-12:
                        current_res = new_res
                        improved = True
                    else:
                        # Revert
                        arr[arr_idx] = orig_val
                        
            if not improved:
                break
                
        if np.allclose(compute_residual(U, V, W), 0.0, atol=1e-10):
            return U, V, W, True
        return U, V, W, False

    start_time = time.time()
    timeout = 145.0  # Leave 5s buffer for safety
    
    # Try ranks from 18 to 29
    for R in range(18, 30):
        if time.time() - start_time > timeout:
            break
            
        n_attempts = 50 if R < 22 else 20
        
        for attempt in range(n_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Random initialization
            # Use different distributions
            if attempt % 2 == 0:
                x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            else:
                x0 = np.random.normal(0, 0.5, R * (nm + mp + pn))
                
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=500)
                
                if result.cost < 1e-3:
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    # Snap to half-integers
                    U_rounded = snap_to_half_integer(U_flat)
                    V_rounded = snap_to_half_integer(V_flat)
                    W_rounded = snap_to_half_integer(W_flat)
                    
                    # Check if snapped solution is valid
                    if verify_half_integer(U_rounded, V_rounded, W_rounded):
                        return {
                            "U": U_rounded.reshape(nm, R).tolist(),
                            "V": V_rounded.reshape(mp, R).tolist(),
                            "W": W_rounded.reshape(pn, R).tolist()
                        }
                        
                    # If not, try local search on the lattice
                    if time.time() - start_time < timeout - 10:
                        U_ls, V_ls, W_ls, success = local_search(U_rounded, V_rounded, W_rounded, R)
                        if success:
                            return {
                                "U": U_ls.reshape(nm, R).tolist(),
                                "V": V_ls.reshape(mp, R).tolist(),
                                "W": W_ls.reshape(pn, R).tolist()
                            }
                        
            except Exception:
                continue
                
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
