import numpy as np
import scipy.optimize
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Search for lower rank
    start_time = time.time()
    time_limit = 110.0  # Leave margin for overhead
    
    # Try ranks from 23 down to 1
    for R in range(23, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts for this rank
        max_restarts = 20 if R > 20 else 50
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            np.random.seed(42 + restart + R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Gauge fixing: set first element of each matrix to 1?
            # This might break symmetry but can cause issues if the solution has 0 there.
            # Let's just use random init for now.
            
            # Flatten
            def unpack(y):
                u_flat = y[:n*m*R]
                v_flat = y[n*m*R : n*m*R + m*p*R]
                w_flat = y[n*m*R + m*p*R :]
                return u_flat.reshape(n*m, R), v_flat.reshape(m*p, R), w_flat.reshape(p*n, R)
                
            def objective(y):
                u, v, w = unpack(y)
                # Compute tensor
                # T_calc[a,b,c] = sum_r u[a,r] v[b,r] w[c,r]
                # Use einsum
                t_calc = np.einsum('ar,br,cr->abc', u, v, w)
                err = t_calc - T
                return np.sum(err * err)
                
            def gradient(y):
                u, v, w = unpack(y)
                t_calc = np.einsum('ar,br,cr->abc', u, v, w)
                err = t_calc - T
                # Grad U: 2 * sum_{b,c} err[a,b,c] v[b,r] w[c,r]
                gu = 2 * np.einsum('abc,br,cr->ar', err, v, w)
                gv = 2 * np.einsum('abc,ar,cr->br', err, u, w)
                gw = 2 * np.einsum('abc,ar,br->cr', err, u, v)
                return np.concatenate([gu.flatten(), gv.flatten(), gw.flatten()])
                
            y0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    objective, 
                    y0, 
                    jac=gradient, 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-6:
                u, v, w = unpack(res.x)
                
                # Snap to half-integers
                u_snap = np.round(2 * u) / 2.0
                v_snap = np.round(2 * v) / 2.0
                w_snap = np.round(2 * w) / 2.0
                
                # Check exactness
                t_check = np.einsum('ar,br,cr->abc', u_snap, v_snap, w_snap)
                if np.allclose(t_check, T, atol=1e-9):
                    # Convert to lists
                    return {
                        "U": u_snap.tolist(),
                        "V": v_snap.tolist(),
                        "W": w_snap.tolist()
                    }
                    
    # Return schoolbook if no better found
    return {
        "U": U_sb,
        "V": V_sb,
        "W": W_sb
    }
