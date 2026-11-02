import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precompute target tensor T
    T = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0
    T_flat = T.flatten()
    
    # Function to compute tensor product
    def tensor_product(u_flat, v_flat, w_flat, r):
        u = u_flat.reshape(n, m, r)
        v = v_flat.reshape(m, p, r)
        w = w_flat.reshape(p, n, r)
        # R[i, j, jp, k, kp, ip]
        R = np.einsum('ijc,jkc,kic->ijkkii', u, v, w, optimize=True)
        return R.flatten()

    # Try ranks lower than schoolbook
    # We try a few ranks, with a few restarts each
    ranks_to_try = [23, 22]
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            np.random.seed(100 + r + restart)
            x0 = np.random.randn(r * (n*m + m*p + p*n))
            
            def loss_func(params, r=r):
                u = params[:n*m]
                v = params[n*m : n*m + m*p]
                w = params[n*m + m*p : n*m + m*p + p*n]
                
                R_flat = tensor_product(u, v, w, r)
                diff = R_flat - T_flat
                return np.dot(diff, diff)
            
            # Use L-BFGS-B
            # Limit iterations to save time
            res = minimize(loss_func, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-6})
            
            if res.fun < 1e-6:
                # Check snap
                u = res.x[:n*m].reshape(n*m, r)
                v = res.x[n*m : n*m + m*p].reshape(m*p, r)
                w = res.x[n*m + m*p : n*m + m*p + p*n].reshape(p*n, r)
                
                u_snap = np.round(u * 2) / 2
                v_snap = np.round(v * 2) / 2
                w_snap = np.round(w * 2) / 2
                
                # Verify
                R_snap_flat = tensor_product(u_snap.flatten(), v_snap.flatten(), w_snap.flatten(), r)
                if np.allclose(R_snap_flat, T_flat, atol=1e-9):
                    best_rank = r
                    best_U = u_snap.tolist()
                    best_V = v_snap.tolist()
                    best_W = w_snap.tolist()
                    break
        else:
            continue
        break
        
    return {"U": best_U, "V": best_V, "W": best_W}
