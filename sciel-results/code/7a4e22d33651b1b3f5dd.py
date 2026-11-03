<<<<<<< SEARCH
def _gradient_descent(n, m, p, r, max_time=10, seed=42):
    """Gradient descent to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, r) * 0.1
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    lr = 0.05
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Compute T
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Compute gradients
        # dT/dU_3d[i,j,c] = V_3d[j,:,c] * W_3d[:,i,c] -> shape (n, m, r)
        # More precisely: dT[i,j,k]/dU[i,j,c] = V[j,k,c] * W[k,i,c]
        # So dL/dU[i,j,c] = sum_k 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c]
        
        # grad_U_3d[i,j,c] = sum_k 2*(T[i,j,k]-1) * V_3d[j,k,c] * W_3d[k,i,c]
        # Using einsum: 2*(T-1)[i,j,k] * V[j,k,c] * W[k,i,c] -> sum over k
        dL_dU_3d = 2 * np.einsum('ijk,jkc,kic->ijc', T - 1.0, V_3d, W_3d)
        
        # grad_V_3d[j,k,c] = sum_i 2*(T[i,j,k]-1) * U[i,j,c] * W[k,i,c]
        dL_dV_3d = 2 * np.einsum('ijk,ijc,kic->jkc', T - 1.0, U_3d, W_3d)
        
        # grad_W_3d[k,i,c] = sum_j 2*(T[i,j,k]-1) * U[i,j,c] * V[j,k,c]
        dL_dW_3d = 2 * np.einsum('ijk,ijc,jkc->kic', T - 1.0, U_3d, V_3d)
        
        # Update
        U -= lr * dL_dU_3d.reshape(n * m, r)
        V -= lr * dL_dV_3d.reshape(m * p, r)
        W -= lr * dL_dW_3d.reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Adaptive learning rate
        lr *= 0.999
        
        if lr < 1e-6:
            break
    
    # Check if we found a good solution
    ok, final_err = _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-8)
    if ok or final_err < 1e-6:
        return best_U, best_V, best_W, r
    
    return None
=======
def _alternating_least_squares(n, m, p, r, max_time=10, seed=42):
    """Alternating least squares to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with structured random
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    # Normalize columns to start
    for c in range(r):
        nu = np.linalg.norm(U[:, c])
        nv = np.linalg.norm(V[:, c])
        nw = np.linalg.norm(W[:, c])
        if nu > 1e-12:
            U[:, c] /= nu
        if nv > 1e-12:
            V[:, c] /= nv
        if nw > 1e-12:
            W[:, c] /= nw
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Compute T
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
        
        # Alternating minimization: fix V, W and solve for U
        # For each (i,j), we want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
        # This is: sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1
        # Let A[i,j,k,c] = V[j,k,c] * W[k,i,c], then sum_c U[i,j,c] * A[i,j,k,c] = 1
        # This is a linear system for U[i,j,:] given fixed V, W
        
        # More efficiently: for fixed V, W, the equation is:
        # For each i, j: the vector (sum_c U[i,j,c] * V[j,k,c] * W[k,i,c])_{k=0}^{p-1} should be all 1s
        
        # Solve for U: For each (i,j), we have p equations in r unknowns U[i,j,0..r-1]
        # But we also need to maintain consistency. Let's use a simpler approach:
        # Minimize ||T - 1||_F^2 over U given V, W
        
        # Gradient descent step for U with line search
        # Actually, let's do proper alternating least squares
        
        # For U: fix V, W. The objective is sum_{i,j,k} (sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] - 1)^2
        # This is quadratic in U. For each (i,j), define b[i,j,k] = 1
        # A[i,j,k,c] = V[j,k,c] * W[k,i,c]
        # Minimize ||A[i,j] @ U[i,j] - b[i,j]||^2
        
        # Build the system efficiently
        # A has shape (n, m, p, r), b has shape (n, m, p)
        A_U = np.einsum('jkc,kic->ijkc', V_3d, W_3d)  # (n, m, p, r)
        b_U = np.ones((n, m, p))
        
        # Solve least squares: for each (i,j), A_U[i,j] @ U[i,j] ≈ b_U[i,j]
        # A_U[i,j] is (p, r), U[i,j] is (r,)
        try:
            U_new = np.zeros((n, m, r))
            for i in range(n):
                for j in range(m):
                    if A_U[i, j].shape[0] >= r:
                        sol, _, _, _ = np.linalg.lstsq(A_U[i, j], b_U[i, j], rcond=None)
                    else:
                        # Underdetermined: use minimum norm solution
                        sol = np.linalg.pinv(A_U[i, j]) @ b_U[i, j]
                    U_new[i, j] = sol
            U = U_new.reshape(n * m, r)
            U_3d = U.reshape(n, m, r)
        except:
            pass
        
        # For V: fix U, W
        A_V = np.einsum('ijc,kic->jkc', U_3d, W_3d)  # (m, p, r)
        b_V = np.ones((m, p))
        
        try:
            V_new = np.zeros((m, p, r))
            for j in range(m):
                for k in range(p):
                    if A_V[j, k].shape[0] >= r:
                        sol, _, _, _ = np.linalg.lstsq(A_V[j, k], b_V[j, k], rcond=None)
                    else:
                        sol = np.linalg.pinv(A_V[j, k]) @ b_V[j, k]
                    V_new[j, k] = sol
            V = V_new.reshape(m * p, r)
            V_3d = V.reshape(m, p, r)
        except:
            pass
        
        # For W: fix U, V
        A_W = np.einsum('ijc,jkc->kic', U_3d, V_3d)  # (p, n, r)
        b_W = np.ones((p, n))
        
        try:
            W_new = np.zeros((p, n, r))
            for k in range(p):
                for i in range(n):
                    if A_W[k, i].shape[0] >= r:
                        sol, _, _, _ = np.linalg.lstsq(A_W[k, i], b_W[k, i], rcond=None)
                    else:
                        sol = np.linalg.pinv(A_W[k, i]) @ b_W[k, i]
                    W_new[k, i] = sol
            W = W_new.reshape(p * n, r)
            W_3d = W.reshape(p, n, r)
        except:
            pass
    
    # Check if we found a good solution
    ok, final_err = _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-8)
    if ok or final_err < 1e-6:
        return best_U, best_V, best_W, r
    
    return None
>>>>>>> REPLACE

