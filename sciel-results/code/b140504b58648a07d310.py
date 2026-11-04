import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140

    # Target: Find rank 16 decomposition of 2x3x3 multiplication tensor
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    def find_decomposition_233(r, max_time=100):
        t0 = time.time()
        rng = np.random.RandomState(42)
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        # Try several random restarts
        for restart in range(50):
            if time.time() - t0 > max_time:
                break
            
            U = rng.randn(n2*m2, r)
            V = rng.randn(m2*p2, r)
            W = rng.randn(p2*n2, r)
            
            # Flatten parameters
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n2*m2*r], x[n2*m2*r:n2*m2*r+m2*p2*r], x[n2*m2*r+m2*p2*r:]
                U_mat = U_flat.reshape(n2*m2, r)
                V_mat = V_flat.reshape(m2*p2, r)
                W_mat = W_flat.reshape(p2*n2, r)
                
                val = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                diff = T2 - val
                return np.sum(diff**2)
            
            # Use BFGS or L-BFGS-B
            # L-BFGS-B is good for medium sized problems
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < best_err:
                x = res.x
                U_mat = x[:n2*m2*r].reshape(n2*m2, r)
                V_mat = x[n2*m2*r:n2*m2*r+m2*p2*r].reshape(m2*p2, r)
                W_mat = x[n2*m2*r+m2*p2*r:].reshape(p2*n2, r)
                
                err = res.fun
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
                    
                    # Try to snap to half-integers
                    # The solution is expected to be integer or half-integer
                    U_snap = np.round(U_mat * 2) / 2
                    V_snap = np.round(V_mat * 2) / 2
                    W_snap = np.round(W_mat * 2) / 2
                    
                    val_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                    err_snap = np.sum((T2 - val_snap)**2)
                    
                    if err_snap < 1e-9:
                        return U_snap, V_snap, W_snap

        if best_err < 1e-9:
            # Final snap check
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            val_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            err_snap = np.sum((T2 - val_snap)**2)
            if err_snap < 1e-9:
                return U_snap, V_snap, W_snap
            else:
                # If not snapped, but error is low, maybe return original? 
                # Task requires exact. If not exact, it's invalid.
                # So we must return None if not exact.
                pass
                
        return None

    res_233 = find_decomposition_233(16)
    
    if res_233 is not None:
        U1, V1, W1 = res_233
        r = 16
        total_rank = 2 * r
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Block 1: columns 0-2 of B (k=0,1,2)
        U[:, 0:r] = U1
        V[0:9, 0:r] = V1
        W[0:6, 0:r] = W1
        
        # Block 2: columns 3-5 of B (k=3,4,5)
        U[:, r:2*r] = U1
        V[9:18, r:2*r] = V1
        W[6:12, r:2*r] = W1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
