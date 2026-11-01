import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 5
    DEADLINE = 145.0
    start_time = time.time()
    
    # Precompute target tensor T of shape (4, 10, 10)
    T = np.zeros((4, 10, 10), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def check_exact(U, V, W, r, tol=1e-5):
        S = np.einsum('ut,vt,wt->uvw', U, V, W)
        return np.allclose(S, T, atol=tol)

    def snap_to_half_integers(arr, tol=1e-4):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def snap_to_integers(arr, tol=1e-4):
        snapped = np.round(arr)
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_als(r, max_iters=500, tol=1e-10, n_restarts=50):
        """Alternating Least Squares for tensor decomposition."""
        best_err = np.inf
        best_UVW = None
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization with various strategies
            if restart % 5 == 0:
                np.random.seed(restart * 31 + r)
                U = np.random.normal(0, 1, (4, r))
                V = np.random.normal(0, 1, (10, r))
                W = np.random.normal(0, 1, (10, r))
            elif restart % 5 == 1:
                np.random.seed(restart * 37 + r)
                U = np.random.choice([-1, 0, 1], (4, r))
                V = np.random.choice([-1, 0, 1], (10, r))
                W = np.random.choice([-1, 0, 1], (10, r))
            elif restart % 5 == 2:
                np.random.seed(restart * 41 + r)
                U = np.random.uniform(-1, 1, (4, r))
                V = np.random.uniform(-1, 1, (10, r))
                W = np.random.uniform(-1, 1, (10, r))
            elif restart % 5 == 3:
                np.random.seed(restart * 43 + r)
                # Sparse initialization
                U = np.zeros((4, r))
                V = np.zeros((10, r))
                W = np.zeros((10, r))
                for t in range(r):
                    U[np.random.randint(0, 4), t] = np.random.choice([-1, 1])
                    V[np.random.randint(0, 10), t] = np.random.choice([-1, 1])
                    W[np.random.randint(0, 10), t] = np.random.choice([-1, 1])
            else:
                np.random.seed(restart * 47 + r)
                U = np.random.normal(0, 0.5, (4, r))
                V = np.random.normal(0, 0.5, (10, r))
                W = np.random.normal(0, 0.5, (10, r))
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > DEADLINE - 15:
                    break
                
                # Update U: fix V, W
                # S[u,v,w] = sum_t U[u,t] V[v,t] W[w,t]
                # For each u: T[u,:,:] = U[u,:] @ (V * W) element-wise in t
                # Reshape: for each t, we have outer product V[:,t] and W[:,t]
                # T[u,v,w] = sum_t U[u,t] * V[v,t] * W[w,t]
                # Let M_t[v,w] = V[v,t] * W[w,t]
                # Then T[u,:,:] = sum_t U[u,t] * M_t
                # So U[u,:] solves: T[u,:,:].reshape(-1) = U[u,:] @ stack(M_t.flatten())
                
                # Build design matrix for U
                # Each column corresponds to a (v,w) pair
                # Design matrix A_U has shape (100, r) where A_U[v*10+w, t] = V[v,t]*W[w,t]
                A_U = np.einsum('vt,wt->vwt', V, W).reshape(100, r)
                b_U = T.reshape(100, 4).T  # shape (4, 100)
                U_new, *_ = np.linalg.lstsq(A_U.T, b_U.T, rcond=None)
                U_new = U_new.T  # shape (4, r)
                
                # Update V
                A_V = np.einsum('ut,wt->uwt', U, W).reshape(100, r)
                b_V = T.reshape(4, 100).T  # shape (100, 4) -> need (10, 100)
                # T[v*10+w, u] for fixed v: T[:, v*10:(v+1)*10, :].reshape(-1)
                V_new = np.zeros((10, r))
                for v in range(10):
                    b_v = T[:, v, :].reshape(100)  # Wait, T is (4,10,10)
                    # T[u, v, w] so for fixed v, we have T[:, v, :] shape (4, 10)
                    b_v = T[:, v, :].reshape(40)
                    A_V_v = A_V.reshape(4, 10, r)[:, v, :].reshape(40, r)
                    V_new[v, :], *_ = np.linalg.lstsq(A_V_v, b_v, rcond=None)
                
                # Update W
                W_new = np.zeros((10, r))
                for w in range(10):
                    b_w = T[:, :, w].reshape(40)  # T[u, v, w] for fixed w
                    A_W_w = np.einsum('ut,vt->uvt', U, V).reshape(4, 10, r)[:, :, w].reshape(40, r)
                    W_new[w, :], *_ = np.linalg.lstsq(A_W_w, b_w, rcond=None)
                
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
                
                if it % 10 == 0 or it == max_iters - 1:
                    S = np.einsum('ut,vt,wt->uvw', U, V, W)
                    err = np.sum((S - T)**2)
                    if err < best_err:
                        best_err = err
                        best_UVW = (U.copy(), V.copy(), W.copy())
                    if err < tol:
                        break
            
            if best_err < tol:
                break
        
        if best_UVW is None:
            return None, np.inf
        
        U, V, W = best_UVW
        return (U, V, W), best_err

    def polish(U, V, W, r):
        """Final high-precision polish using scipy minimize."""
        def error(params):
            U_p = params[:4*r].reshape((4, r))
            V_p = params[4*r:14*r].reshape((10, r))
            W_p = params[14*r:].reshape((10, r))
            S = np.einsum('ut,vt,wt->uvw', U_p, V_p, W_p)
            return np.sum((S - T)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        res = minimize(error, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
        
        if res.fun < 1e-10:
            U_p = res.x[:4*r].reshape((4, r))
            V_p = res.x[4*r:14*r].reshape((10, r))
            W_p = res.x[14*r:].reshape((10, r))
            
            # Try snapping
            for snap_func in [snap_to_half_integers, snap_to_integers]:
                U_s = snap_func(U_p)
                V_s = snap_func(V_p)
                W_s = snap_func(W_p)
                if U_s is not None and V_s is not None and W_s is not None:
                    if check_exact(U_s, V_s, W_s, r, tol=1e-5):
                        return (U_s, V_s, W_s)
            
            # Check if raw is close enough
            if check_exact(U_p, V_p, W_p, r, tol=1e-5):
                return (U_p, V_p, W_p)
        
        return None

    def try_find_decomposition(r):
        """Try to find exact rank-r decomposition using ALS + polish."""
        if time.time() > DEADLINE - 20:
            return None
        
        # First try ALS
        result, err = try_als(r, max_iters=300, tol=1e-10, n_restarts=30)
        
        if result is not None and err < 1e-8:
            U, V, W = result
            # Polish
            polished = polish(U, V, W, r)
            if polished is not None:
                return polished
            # Check if raw works
            if check_exact(U, V, W, r, tol=1e-5):
                return (U, V, W)
        
        # If ALS didn't work well, try direct optimization with more restarts
        if time.time() < DEADLINE - 25:
            num_vars = 24 * r
            max_restarts = 50
            
            for restart in range(max_restarts):
                if time.time() > DEADLINE - 10:
                    return None
                
                if restart % 3 == 0:
                    np.random.seed(restart * 7 + r)
                    x0 = np.random.uniform(-1, 1, size=num_vars)
                elif restart % 3 == 1:
                    np.random.seed(restart * 13 + r)
                    x0 = np.random.choice([-1, 0, 1], size=num_vars)
                else:
                    np.random.seed(restart * 17 + r)
                    x0 = np.random.normal(0, 1, size=num_vars)
                
                def error(params):
                    if time.time() > DEADLINE - 5:
                        return 1e10
                    U_p = params[:4*r].reshape((4, r))
                    V_p = params[4*r:14*r].reshape((10, r))
                    W_p = params[14*r:].reshape((10, r))
                    S = np.einsum('ut,vt,wt->uvw', U_p, V_p, W_p)
                    return np.sum((S - T)**2)
                
                res = minimize(error, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.fun < 1e-8:
                    U_p = res.x[:4*r].reshape((4, r))
                    V_p = res.x[4*r:14*r].reshape((10, r))
                    W_p = res.x[14*r:].reshape((10, r))
                    
                    for snap_func in [snap_to_half_integers, snap_to_integers]:
                        U_s = snap_func(U_p)
                        V_s = snap_func(V_p)
                        W_s = snap_func(W_p)
                        if U_s is not None and V_s is not None and W_s is not None:
                            if check_exact(U_s, V_s, W_s, r, tol=1e-5):
                                return (U_s, V_s, W_s)
                    
                    if check_exact(U_p, V_p, W_p, r, tol=1e-5):
                        return (U_p, V_p, W_p)
        
        return None

    # Try decreasing ranks
    for r in [17, 16, 15, 14]:
        if time.time() > DEADLINE - 30:
            break
        decomp = try_find_decomposition(r)
        if decomp:
            U, V, W = decomp
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 18
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(10)]
        W = [[0] * rank for _ in range(10)]
        
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*5 + k0][t_offset+0] = 1
            V[1*5 + k1][t_offset+0] = 1
            V[0*5 + k0][t_offset+1] = 1
            V[0*5 + k1][t_offset+2] = 1
            V[1*5 + k1][t_offset+2] = -1
            V[1*5 + k0][t_offset+3] = 1
            V[0*5 + k0][t_offset+3] = -1
            V[1*5 + k1][t_offset+4] = 1
            V[0*5 + k0][t_offset+5] = 1
            V[0*5 + k1][t_offset+5] = 1
            V[1*5 + k0][t_offset+6] = 1
            V[1*5 + k1][t_offset+6] = 1
            
            W[k0*2 + 0][t_offset+0] = 1
            W[k1*2 + 1][t_offset+0] = 1
            W[k0*2 + 1][t_offset+1] = 1
            W[k1*2 + 1][t_offset+1] = -1
            W[k1*2 + 0][t_offset+2] = 1
            W[k1*2 + 1][t_offset+2] = 1
            W[k0*2 + 0][t_offset+3] = 1
            W[k0*2 + 1][t_offset+3] = 1
            W[k0*2 + 0][t_offset+4] = -1
            W[k1*2 + 0][t_offset+4] = 1
            W[k1*2 + 1][t_offset+5] = 1
            W[k0*2 + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*5 + 4][t] = 1
        W[4*2 + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*5 + 4][t] = 1
        W[4*2 + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*5 + 4][t] = 1
        W[4*2 + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*5 + 4][t] = 1
        W[4*2 + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    return get_rank_18_solution()
