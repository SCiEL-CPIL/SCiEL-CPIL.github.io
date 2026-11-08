import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def check_error(U, V, W):
        """Compute ||sum_r U[u,r]V[v,r]W[w,r] - T||"""
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(Rec - T)
    
    def optimize(U, V, W, max_iter=50, tol=1e-10, deadline=0):
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5: # Safety margin
                break
                
            # Update W
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, R)
            A_W = P_UV.reshape(nm * mp, R)
            
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W_new[w] = sol
                except np.linalg.LinAlgError:
                    W_new[w] = 0.0
            W = W_new
            
            # Update V
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, R)
            A_V = P_UW.reshape(nm * pn, R)
            
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V_new[v] = sol
                except np.linalg.LinAlgError:
                    V_new[v] = 0.0
            V = V_new
            
            # Update U
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, R)
            A_U = P_VW.reshape(mp * pn, R)
            
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U_new[u] = sol
                except np.linalg.LinAlgError:
                    U_new[u] = 0.0
            U = U_new
            
            # Check error
            err = check_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
                
        return best_U, best_V, best_W, best_err

    deadline = time.time() + 140  # 10s safety margin
    
    best_solution = None
    best_rank = n * m * p  # Fallback to schoolbook rank 48
    best_err = float('inf')
    
    # Build schoolbook decomposition
    U_school = np.zeros((nm, n * m * p))
    V_school = np.zeros((mp, n * m * p))
    W_school = np.zeros((pn, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    # Strategy 1: Try to reduce schoolbook rank by 1 (Rank 47)
    # Initialize with schoolbook, zero out one column, and optimize.
    R_target = n * m * p - 1 # 47
    
    # Try zeroing out each column of the schoolbook decomposition
    for zero_col in range(n * m * p):
        if time.time() > deadline - 10:
            break
            
        # Remove the zero column to get R_target columns
        mask = np.ones(n * m * p, dtype=bool)
        mask[zero_col] = False
        U_init = U_school[:, mask]
        V_init = V_school[:, mask]
        W_init = W_school[:, mask]
        
        # Add small random perturbation to break symmetry
        U_init = U_init + np.random.randn(nm, R_target) * 1e-6
        V_init = V_init + np.random.randn(mp, R_target) * 1e-6
        W_init = W_init + np.random.randn(pn, R_target) * 1e-6
        
        U_opt, V_opt, W_opt, err = optimize(U_init, V_init, W_init, max_iter=30, tol=1e-10, deadline=deadline)
        
        if err < best_err:
            best_err = err
            best_solution = (U_opt, V_opt, W_opt)
            best_rank = R_target
            
        if err < 1e-8:
            break

    # Strategy 2: Random restarts for Rank 47 and 46
    if best_err > 1e-8:
        for R_target in [47, 46]:
            if time.time() > deadline - 20:
                break
            max_restarts = 10
            for restart in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                U = np.random.randn(nm, R_target)
                V = np.random.randn(mp, R_target)
                W = np.random.randn(pn, R_target)
                
                U_opt, V_opt, W_opt, err = optimize(U, V, W, max_iter=20, tol=1e-10, deadline=deadline)
                
                if err < best_err:
                    best_err = err
                    best_solution = (U_opt, V_opt, W_opt)
                    best_rank = R_target
                    
                if err < 1e-8:
                    break

    if best_solution is None or best_err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = U_school
        V = V_school
        W = W_school
        best_solution = (U, V, W)
        best_rank = rank

    U, V, W = best_solution
    
    # Convert to list of lists
    def to_list(matrix):
        L = []
        for row in matrix:
            r = []
            for val in row:
                # Check if close to half-integer
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-6:
                    r.append(nearest)
                else:
                    r.append(float(val))
            L.append(r)
        return L

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
