import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Construct the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Helper function to check if a solution is valid after rounding
    def try_round_and_check(U, V, W, T, tol=1e-9):
        # Try half-integer rounding
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_recon = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r, optimize=True)
        if np.linalg.norm(T_recon - T) < tol:
            return (U_r, V_r, W_r)
        
        # Try integer rounding
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        T_recon = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r, optimize=True)
        if np.linalg.norm(T_recon - T) < tol:
            return (U_r, V_r, W_r)
        
        # Try rounding to nearest 0.25
        U_r = np.round(U * 4) / 4
        V_r = np.round(V * 4) / 4
        W_r = np.round(W * 4) / 4
        T_recon = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r, optimize=True)
        if np.linalg.norm(T_recon - T) < tol:
            return (U_r, V_r, W_r)
        
        return None

    # Helper function to do ALS optimization with adaptive regularization
    def als_optimize(U, V, W, T, R, max_iters=200, initial_lam=1e-8, final_lam=1e-12, time_limit=140.0, start_time=None):
        if start_time is None:
            start_time = time.time()
        
        lam = initial_lam
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Decay regularization
            progress = it / max_iters
            lam = initial_lam * (final_lam / initial_lam) ** progress
            
            I_R = np.eye(R) * lam
            
            # Update U
            VtV = V.T @ V
            WtW = W.T @ W
            G_U = VtV * WtW + I_R
            R_U = np.einsum('vc,wc,uvw->uc', V, W, T, optimize=True)
            try:
                U_new = np.linalg.solve(G_U, R_U.T).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(G_U, R_U.T, rcond=None)[0].T
            U = U_new
            
            # Update V
            UtU = U.T @ U
            G_V = UtU * WtW + I_R
            R_V = np.einsum('uc,wc,uvw->vc', U, W, T, optimize=True)
            try:
                V_new = np.linalg.solve(G_V, R_V.T).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(G_V, R_V.T, rcond=None)[0].T
            V = V_new
            
            # Update W
            VtV = V.T @ V
            G_W = UtU * VtV + I_R
            R_W = np.einsum('uc,vc,uvw->wc', U, V, T, optimize=True)
            try:
                W_new = np.linalg.solve(G_W, R_W.T).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(G_W, R_W.T, rcond=None)[0].T
            W = W_new
            
            # Check error periodically
            if it % 5 == 0 or it == max_iters - 1:
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                error = np.linalg.norm(T_recon - T)
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-7:
                    # Try to round and check
                    result = try_round_and_check(U, V, W, T)
                    if result is not None:
                        return result
                    break
        
        # Final polish: run more iterations with very small regularization
        if best_error < 1e-5 and time.time() - start_time < time_limit - 10:
            U, V, W = best_U, best_V, best_W
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                lam = 1e-12
                I_R = np.eye(R) * lam
                
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW + I_R
                R_U = np.einsum('vc,wc,uvw->uc', V, W, T, optimize=True)
                try:
                    U_new = np.linalg.solve(G_U, R_U.T).T
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(G_U, R_U.T, rcond=None)[0].T
                U = U_new
                
                UtU = U.T @ U
                G_V = UtU * WtW + I_R
                R_V = np.einsum('uc,wc,uvw->vc', U, W, T, optimize=True)
                try:
                    V_new = np.linalg.solve(G_V, R_V.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(G_V, R_V.T, rcond=None)[0].T
                V = V_new
                
                VtV = V.T @ V
                G_W = UtU * VtV + I_R
                R_W = np.einsum('uc,vc,uvw->wc', U, V, T, optimize=True)
                try:
                    W_new = np.linalg.solve(G_W, R_W.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(G_W, W.T, rcond=None)[0].T
                W = W_new
            
            result = try_round_and_check(U, V, W, T, tol=1e-10)
            if result is not None:
                return result
        
        return None

    # Try ranks from lower to higher, starting with promising values
    # Based on Strassen-like bounds, try ranks around 100-120
    ranks_to_try = list(range(100, 121))
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Multiple restarts with different initializations
        for restart in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            
            seed = 42 + restart * 1000 + R
            np.random.seed(seed)
            
            # Better initialization: use structure-aware random init
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(dim_u, R) * scale
            V = np.random.randn(dim_v, R) * scale
            W = np.random.randn(dim_w, R) * scale
            
            # Add some structure: initialize based on the tensor pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        # Add small bias to encourage structure
                        if restart == 0:
                            U[u, :] += 0.01
                            V[v, :] += 0.01
                            W[w, :] += 0.01
            
            result = als_optimize(U, V, W, T, R, max_iters=150, 
                                  initial_lam=1e-6, final_lam=1e-10,
                                  time_limit=time_limit, start_time=start_time)
            
            if result is not None:
                U_r, V_r, W_r = result
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
