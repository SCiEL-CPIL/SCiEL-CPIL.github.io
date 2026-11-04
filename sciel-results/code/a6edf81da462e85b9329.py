import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_index_arrays(n, m, p):
    """Precompute index arrays for vectorized objective/gradient."""
    u_idx = []
    v_idx = []
    w_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx.append(i * m + j)
                v_idx.append(j * p + k)
                w_idx.append(k * n + i)
    return np.array(u_idx), np.array(v_idx), np.array(w_idx)

def _optimize_rank_343(r, time_limit=20):
    """
    Optimizes for rank r decomposition of 3x4x3 matrix multiplication.
    Returns (U, V, W, success)
    """
    n, m, p = 3, 4, 3
    start_time = time.time()
    u_idx, v_idx, w_idx = _build_index_arrays(n, m, p)
    total_vars = n*m*r + m*p*r + p*n*r
    
    def unpack(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        return U, V, W

    def objective(x):
        U, V, W = unpack(x)
        # Vectorized: for each (i,j,k) term, compute dot(U[u_idx], V[v_idx]*W[w_idx])
        U_sel = U[u_idx, :]  # (nm, r)
        V_sel = V[v_idx, :]  # (nm, r)
        W_sel = W[w_idx, :]  # (nm, r)
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (nm,)
        err = np.sum((vals - 1.0)**2)
        return err

    def gradient(x):
        U, V, W = unpack(x)
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        diff = 2.0 * (vals - 1.0)  # (nm,)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Accumulate gradients
        np.add.at(grad_U, (u_idx, np.arange(r)), diff[:, None] * (V_sel * W_sel))
        np.add.at(grad_V, (v_idx, np.arange(r)), diff[:, None] * (U_sel * W_sel))
        np.add.at(grad_W, (w_idx, np.arange(r)), diff[:, None] * (U_sel * V_sel))
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_loss = float('inf')
    best_x = None
    
    # Build schoolbook-based initialization
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    # Truncate to rank r (take first r columns)
    U_init = U_school[:, :r].copy()
    V_init = V_school[:, :r].copy()
    W_init = W_school[:, :r].copy()
    
    num_inits = 15
    for init_idx in range(num_inits):
        if time.time() - start_time > time_limit:
            break
        
        if init_idx == 0:
            # Schoolbook truncated with tiny perturbation
            x0 = np.concatenate([
                (U_init + 0.01 * np.random.randn(n*m, r)).flatten(),
                (V_init + 0.01 * np.random.randn(m*p, r)).flatten(),
                (W_init + 0.01 * np.random.randn(p*n, r)).flatten()
            ])
        elif init_idx == 1:
            # Schoolbook truncated with larger perturbation
            x0 = np.concatenate([
                (U_init + 0.1 * np.random.randn(n*m, r)).flatten(),
                (V_init + 0.1 * np.random.randn(m*p, r)).flatten(),
                (W_init + 0.1 * np.random.randn(p*n, r)).flatten()
            ])
        elif init_idx == 2:
            # Pure random small
            x0 = np.random.randn(total_vars) * 0.05
        elif init_idx == 3:
            # Random with structure: each row mostly zeros with one nonzero
            U_rand = np.zeros((n*m, r))
            V_rand = np.zeros((m*p, r))
            W_rand = np.zeros((p*n, r))
            for col in range(r):
                U_rand[np.random.randint(n*m), col] = np.random.choice([-1, 1])
                V_rand[np.random.randint(m*p), col] = np.random.choice([-1, 1])
                W_rand[np.random.randint(p*n), col] = np.random.choice([-1, 1])
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        else:
            # Mix of schoolbook and random
            alpha = 0.5
            x0 = np.concatenate([
                (alpha * U_init + (1-alpha) * np.random.randn(n*m, r) * 0.1).flatten(),
                (alpha * V_init + (1-alpha) * np.random.randn(m*p, r) * 0.1).flatten(),
                (alpha * W_init + (1-alpha) * np.random.randn(p*n, r) * 0.1).flatten()
            ])

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try snapping to half-integers
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap, V_snap, W_snap, True
                    
                # Also try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-6):
                    return U_int, V_int, W_int, True
                    
        except Exception:
            pass
            
    # Polish pass: if we have a good solution, refine it
    if best_x is not None and best_loss < 0.1:
        try:
            from scipy.optimize import minimize
            res2 = minimize(objective, best_x, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res2.fun < best_loss:
                best_loss = res2.fun
                best_x = res2.x.copy()
                
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap, V_snap, W_snap, True
                    
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-6):
                    return U_int, V_int, W_int, True
        except Exception:
            pass
            
    if best_x is not None and best_loss < 1e-4:
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
            return U_snap, V_snap, W_snap, True
            
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-5):
            return U_int, V_int, W_int, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Strategy: Decompose 3x4x6 into two 3x4x3 blocks.
    # If we find a rank r < 36 for 3x4x3, we can form a rank 2r < 72 solution for 3x4x6.
    
    # Try to find low rank for 3x4x3
    # Start from a reasonable guess. Schoolbook is 36.
    # Try ranks from 30 upward.
    for r in range(30, 36):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        
        U_343, V_343, W_343, success = _optimize_rank_343(r, time_limit=min(remaining, 30))
        
        if success:
            # Construct 3x4x6 solution
            # U_new: (12, 2r)
            # V_new: (24, 2r)
            # W_new: (18, 2r)
            
            U_new = np.zeros((n*m, 2*r))
            V_new = np.zeros((m*p, 2*r))
            W_new = np.zeros((p*n, 2*r))
            
            # Block 1: k=0,1,2
            U_new[:, 0:r] = U_343
            V_new[0:12, 0:r] = V_343 # m*p for 3x4x3 is 4*3=12
            W_new[0:9, 0:r] = W_343   # p*n for 3x4x3 is 3*3=9
            
            # Block 2: k=3,4,5
            U_new[:, r:2*r] = U_343
            
            for j in range(m):
                for k_prime in range(3):
                    V_new[j*6 + 3 + k_prime, r:2*r] = V_343[j*3 + k_prime, :]
                    
            for k_prime in range(3):
                for i in range(n):
                    W_new[(3+k_prime)*3 + i, r:2*r] = W_343[k_prime*3 + i, :]
            
            # Verify the full 3x4x6 decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U_new]
                V_list = [[float(x) for x in row] for row in V_new]
                W_list = [[float(x) for x in row] for row in W_new]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
