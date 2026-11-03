import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_error(U, V, W):
        """
        Compute the squared error between the target tensor and the decomposition.
        """
        R = U.shape[1]
        D = np.zeros_like(T)
        for r in range(R):
            D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.sum((T - D) ** 2)

    def verify(U, V, W, tol=1e-9):
        """
        Check if the decomposition is exact and entries are half-integers.
        """
        R = U.shape[1]
        D = np.zeros_like(T)
        for r in range(R):
            D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if np.max(np.abs(T - D)) > tol:
            return False
        
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def snap_to_half_integers(U, V, W):
        """
        Snap values to the nearest half-integer.
        """
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        return U_snapped, V_snapped, W_snapped

    def als_decomposition(target_rank, max_iters=200, tol=1e-10):
        """
        Perform Alternating Least Squares to find a real-valued decomposition.
        """
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        prev_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            # For each u, minimize || T[u,:,:] - sum_r U[u,r] (V[:,r] outer W[:,r]) ||^2
            # This is a linear least squares problem: U[u, :] @ X_u = T[u, :].flatten()
            # X_u has shape (m*p*p*n, target_rank) where X_u[:, r] = (V[:, r] outer W[:, r]).flatten()
            
            # Precompute X_u for all u? 
            # X_u depends on V and W.
            # Shape of V is (m*p, R), W is (p*n, R).
            # Outer product V[:, r] outer W[:, r] has shape (m*p, p*n).
            # We can construct a matrix X of shape (n*m, m*p*p*n, R) efficiently?
            # Or just loop over u. n*m = 9. Small.
            
            for u in range(n * m):
                # Construct X_u: (m*p*p*n, R)
                # X_u[ (v, w), r ] = V[v, r] * W[w, r]
                # Vectorize v, w.
                X_u = np.zeros((m * p * p * n, target_rank))
                idx = 0
                for v in range(m * p):
                    for w in range(p * n):
                        for r in range(target_rank):
                            X_u[idx, r] = V[v, r] * W[w, r]
                        idx += 1
                
                # Target vector T_u: (m*p*p*n,)
                T_u = T[u, :, :].flatten()
                
                # Solve U[u, :] X_u^T = T_u^T  =>  X_u U[u, :]^T = T_u
                # U[u, :] is (1, R). X_u is (N, R).
                # We want min || T_u - X_u U_u^T ||
                # This is min || T_u^T - U_u X_u^T ||
                # Standard least squares: X_u @ U_u^T = T_u
                # U_u^T = np.linalg.lstsq(X_u, T_u, rcond=None)[0]
                # U_u = U_u^T.T
                
                try:
                    U_u = np.linalg.lstsq(X_u, T_u, rcond=None)[0]
                except:
                    U_u = np.zeros(target_rank)
                    
                U[u, :] = U_u
            
            # Solve for V
            # Symmetric logic
            for v in range(m * p):
                X_v = np.zeros((n * m * p * n, target_rank))
                idx = 0
                for u in range(n * m):
                    for w in range(p * n):
                        for r in range(target_rank):
                            X_v[idx, r] = U[u, r] * W[w, r]
                        idx += 1
                
                T_v = T[:, v, :].flatten()
                
                try:
                    V_v = np.linalg.lstsq(X_v, T_v, rcond=None)[0]
                except:
                    V_v = np.zeros(target_rank)
                    
                V[v, :] = V_v

            # Solve for W
            for w in range(p * n):
                X_w = np.zeros((n * m * m * p, target_rank))
                idx = 0
                for u in range(n * m):
                    for v in range(m * p):
                        for r in range(target_rank):
                            X_w[idx, r] = U[u, r] * V[v, r]
                        idx += 1
                
                T_w = T[:, :, w].flatten()
                
                try:
                    W_w = np.linalg.lstsq(X_w, T_w, rcond=None)[0]
                except:
                    W_w = np.zeros(target_rank)
                    
                W[w, :] = W_w

            # Compute error
            error = compute_error(U, V, W)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        return U, V, W, error

    def local_search(U_init, V_init, W_init, target_rank, max_iters=5000):
        """
        Perform local search to minimize error, starting from a good real-valued estimate.
        """
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        # Snap to half-integers first
        U_cur, V_cur, W_cur = snap_to_half_integers(U_cur, V_cur, W_cur)
        
        current_error = compute_error(U_cur, V_cur, W_cur)
        
        rows = [n * m, m * p, p * n]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if current_error < 1e-9:
                if verify(U_cur, V_cur, W_cur):
                    return U_cur, V_cur, W_cur, current_error

            # Pick a random matrix and element
            mat_idx = random.choice([0, 1, 2])
            row = random.randint(0, rows[mat_idx] - 1)
            col = random.randint(0, target_rank - 1)
            
            # Store old value
            if mat_idx == 0:
                old_val = U_cur[row, col]
                U_cur[row, col] += random.choice([-1.0, -0.5, 0.5, 1.0])
            elif mat_idx == 1:
                old_val = V_cur[row, col]
                V_cur[row, col] += random.choice([-1.0, -0.5, 0.5, 1.0])
            else:
                old_val = W_cur[row, col]
                W_cur[row, col] += random.choice([-1.0, -0.5, 0.5, 1.0])
            
            new_error = compute_error(U_cur, V_cur, W_cur)
            
            if new_error < current_error:
                current_error = new_error
            else:
                # Revert
                if mat_idx == 0:
                    U_cur[row, col] = old_val
                elif mat_idx == 1:
                    V_cur[row, col] = old_val
                else:
                    W_cur[row, col] = old_val
        
        return U_cur, V_cur, W_cur, current_error

    # Try ranks from 25 to 35
    best_U = None
    best_V = None
    best_W = None
    best_rank = 36
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, 36))
    V_school = np.zeros((m * p, 36))
    W_school = np.zeros((p * n, 36))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    for target_rank in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        
        found = False
        num_restarts = 0
        while time.time() < deadline - 5 and not found:
            num_restarts += 1
            
            # Use ALS to get a good real-valued initialization
            U_als, V_als, W_als, err_als = als_decomposition(target_rank, max_iters=50)
            
            # If ALS found a near-exact solution, snap and verify
            if err_als < 1e-6:
                U_snap, V_snap, W_snap = snap_to_half_integers(U_als, V_als, W_als)
                if verify(U_snap, V_snap, W_snap):
                    best_U = U_snap.copy()
                    best_V = V_snap.copy()
                    best_W = W_snap.copy()
                    best_rank = target_rank
                    found = True
                    break
            
            # If not, use ALS result as seed for local search
            U_res, V_res, W_res, err = local_search(U_als, V_als, W_als, target_rank)
            
            if err < 1e-9:
                if verify(U_res, V_res, W_res):
                    best_U = U_res.copy()
                    best_V = V_res.copy()
                    best_W = W_res.copy()
                    best_rank = target_rank
                    found = True
                    break

    if best_U is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
