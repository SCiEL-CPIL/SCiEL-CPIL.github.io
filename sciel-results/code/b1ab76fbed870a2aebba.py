import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n, m, p) with all ones
    T = np.ones((n, m, p))
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    start_time = time.time()
    deadline = 115.0 # Leave 5s margin for finalization
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, T, atol=1e-9):
        # U, V, W are expected to be flattened (n*m, r), etc.
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        try:
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False, None, None, None
            
        if np.allclose(T_approx, T, atol=atol):
            return True, U, V, W
        return False, None, None, None

    # Helper to round to half-integers
    def round_to_half(x):
        return np.round(x * 2) / 2.0

    # Vectorized ALS solver for fixed rank r
    def solve_als(r, max_iter=100, tol=1e-10, seed=0):
        np.random.seed(seed)
        # Initialize randomly
        U = np.random.randn(n, m, r)
        V = np.random.randn(m, p, r)
        W = np.random.randn(p, n, r)
        
        # Normalize
        for c in range(r):
            norm_U = np.linalg.norm(U[:, :, c])
            norm_V = np.linalg.norm(V[:, :, c])
            norm_W = np.linalg.norm(W[:, :, c])
            if norm_U > 0: U[:, :, c] /= norm_U
            if norm_V > 0: V[:, :, c] /= norm_V
            if norm_W > 0: W[:, :, c] /= norm_W
            
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None, None, None
                
            # Update U
            # M_U[i, j, k, c] = V[j, k, c] * W[k, i, c]
            M_U = np.einsum('jkc,kic->ijkc', V, W)
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_U[i, j], T[i, j], rcond=None)
                        new_U[i, j] = sol
                    except Exception:
                        new_U[i, j] = U[i, j]
            U = new_U
            
            # Update V
            # M_V[j, k, i, c] = U[i, j, c] * W[k, i, c]
            M_V = np.einsum('ijc,kic->jkic', U, W)
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_V[j, k], T[:, j, k], rcond=None)
                        new_V[j, k] = sol
                    except Exception:
                        new_V[j, k] = V[j, k]
            V = new_V
            
            # Update W
            # M_W[k, i, j, c] = U[i, j, c] * V[j, k, c]
            M_W = np.einsum('ijc,jkc->kijc', U, V)
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_W[k, i], T[i, :, k], rcond=None)
                        new_W[k, i] = sol
                    except Exception:
                        new_W[k, i] = W[k, i]
            W = new_W
            
            # Check convergence
            T_approx = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = np.linalg.norm(T_approx - T)
            if err < tol or (err > prev_err and it > 10):
                break
            prev_err = err
                
        return U, V, W

    # Search for minimal rank
    # Start from a reasonable lower bound (e.g., 40) up to schoolbook
    # We want the minimal rank, so we should try low ranks first.
    # However, finding a solution for low ranks is harder.
    # Let's try ranks from 35 to 60.
    
    best_found = False
    
    for r in range(35, 61):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts
        found_for_rank = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            U, V, W = solve_als(r, max_iter=50, seed=attempt)
            if U is None:
                continue
                
            # Flatten for is_valid
            U_flat = U.reshape(n * m, r)
            V_flat = V.reshape(m * p, r)
            W_flat = W.reshape(p * n, r)
            
            # Check if valid (real-valued)
            valid, U_v, V_v, W_v = is_valid(U_flat, V_flat, W_flat, T)
            if valid:
                # Round to half-integers
                U_r = round_to_half(U_v)
                V_r = round_to_half(V_v)
                W_r = round_to_half(W_v)
                
                # Verify rounded version
                valid_r, U_r_v, V_r_v, W_r_v = is_valid(U_r, V_r, W_r, T)
                if valid_r:
                    best_rank = r
                    best_U = U_r_v
                    best_V = V_r_v
                    best_W = W_r_v
                    best_found = True
                    break
        
        if best_found:
            break

    if not best_found:
        # Fallback to schoolbook if no better found
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
