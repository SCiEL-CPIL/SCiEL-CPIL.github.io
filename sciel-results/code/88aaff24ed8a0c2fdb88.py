"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    
    # The target tensor T is sparse.
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i, j, k.
    # Other entries are 0.
    
    # We represent the tensor as a 3D array of shape (n*m, m*p, p*n)
    # T[idx_u, idx_v, idx_w]
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def error_func(params, R):
        U = params[:R*n*m].reshape((n*m, R))
        V = params[R*n*m:2*R*n*m].reshape((m*p, R))
        W = params[2*R*n*m:3*R*n*m].reshape((p*n, R))
        
        # Reconstruct T
        # T_recon[idx_u, idx_v, idx_w] = sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
        # We can use einsum: 'ur, vr, wr -> uvw'
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        err = np.sum((T_recon - T)**2)
        return err

    best_result = None
    best_rank = None
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Try ranks from 20 to 26
    for R in range(20, 27):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            # Initialize with small random values
            params0 = np.random.randn(3*R*n*m) * 0.1
            
            # Use L-BFGS-B
            try:
                result = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    U = result.x[:R*n*m].reshape((n*m, R))
                    V = result.x[R*n*m:2*R*n*m].reshape((m*p, R))
                    W = result.x[2*R*n*m:3*R*n*m].reshape((p*n, R))
                    
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snapped
                    T_recon_snap = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(T_recon_snap, T, atol=1e-6):
                        best_result = (U_snap, V_snap, W_snap)
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    # Fallback to schoolbook rank 27
    if best_result is None:
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u, col] = 1
                    V[idx_v, col] = 1
                    W[idx_w, col] = 1
                    col += 1
        U, V, W = U, V, W
    else:
        U, V, W = best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
