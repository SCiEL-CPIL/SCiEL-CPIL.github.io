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

def _optimize_rank_343(r, time_limit=20):
    """
    Optimizes for rank r decomposition of 3x4x3 matrix multiplication.
    Returns (U, V, W, success)
    """
    n, m, p = 3, 4, 3
    start_time = time.time()
    
    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = 2 * (val - 1.0)
                    
                    grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_loss = float('inf')
    best_x = None
    
    # Try a few initializations
    for init_idx in range(10):
        if time.time() - start_time > time_limit:
            break
            
        if init_idx == 0:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1
        elif init_idx == 1:
            x0 = np.zeros(n*m*r + m*p*r + p*n*r)
            num_nonzero = r * 3
            indices = np.random.choice(len(x0), num_nonzero, replace=False)
            x0[indices] = np.random.choice([-1, 1], num_nonzero)
        else:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
                
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap, V_snap, W_snap, True
                    
        except Exception:
            pass
            
    if best_x is not None and best_loss < 1e-6:
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Decompose 3x4x6 into two 3x4x3 blocks.
    # If we find a rank r < 36 for 3x4x3, we can form a rank 2r < 72 solution for 3x4x6.
    
    # Try to find low rank for 3x4x3
    # Start from a reasonable guess. Schoolbook is 36.
    # Try ranks 30 to 35.
    for r in range(30, 36):
        if time.time() > deadline - 10:
            break
        
        U_343, V_343, W_343, success = _optimize_rank_343(r, time_limit=10)
        
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
            # We reuse the same U, V, W structure for the second block of B and C
            # Note: The indices for the second block in V are j*p + (3+k') = j*6 + 3 + k'
            # In the 3x4x3 solution, V index is j*3 + k'.
            # So V_new[j*6 + 3 + k', r + t] = V_343[j*3 + k', t]
            # And W_new[(3+k')*3 + i, r + t] = W_343[k'*3 + i, t]
            
            U_new[:, r:2*r] = U_343
            
            for j in range(m):
                for k_prime in range(3):
                    # V index in new: j*6 + 3 + k_prime
                    # V index in old: j*3 + k_prime
                    V_new[j*6 + 3 + k_prime, r:2*r] = V_343[j*3 + k_prime, :]
                    
            for k_prime in range(3):
                for i in range(n):
                    # W index in new: (3+k_prime)*3 + i
                    # W index in old: k_prime*3 + i
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
