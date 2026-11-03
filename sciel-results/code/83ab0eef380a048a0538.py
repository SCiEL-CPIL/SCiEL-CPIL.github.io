import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C
    # T_target[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-8):
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)

    def try_simultaneous_linear(r, max_iter=50, n_restarts=10, seed=42):
        """
        Simultaneous linear system approach:
        - Initialize all three factors randomly
        - Alternate: fix one factor, solve linear system for the other two
        - This is fundamentally different from peeling (all r terms refined together)
        """
        np.random.seed(seed)
        
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        best_residual = np.inf
        best_solution = None
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 8:
                break
            
            # Normalize factors
            for s in range(r):
                nU = np.linalg.norm(U[:, s])
                nV = np.linalg.norm(V[:, s])
                nW = np.linalg.norm(W[:, s])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    U[:, s] /= nU
                    V[:, s] /= nV
                    W[:, s] /= nW
            
            # Compute current residual
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            residual = T_target - T_current
            res_norm = np.linalg.norm(residual)
            
            if res_norm < best_residual:
                best_residual = res_norm
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if res_norm < 1e-8:
                break
            
            # Alternate: fix W, solve for U and V
            # T_target[u,v,w] = sum_s U[u,s]*V[v,s]*W[w,s]
            # For fixed W, this is linear in U and V
            # Rearrange: for each w, T_target[:,:,w] = sum_s (U*V)[:,:,s] * W[w,s]
            # Let M_s = U[:,s] * V[:,s] (outer product), then T_target[:,:,w] = sum_s M_s * W[w,s]
            
            # Solve for U and V given W:
            # This is a bilinear system. Use alternating minimization within this step.
            # Fix V, solve for U: T_target[u,v,w] = sum_s U[u,s] * (V[v,s]*W[w,s])
            # For each u: T_target[u,v,w] over v,w = sum_s U[u,s] * sum_{v,w} V[v,s]*W[w,s] * delta
            # Actually, let's use a different approach.
            
            # Fix V and W, solve for U:
            # T_target[u,v,w] = sum_s U[u,s] * V[v,s] * W[w,s]
            # For each u: sum_{v,w} T_target[u,v,w] * V[v,s'] * W[w,s'] = U[u,s'] * sum_{v,w} V[v,s']*V[v,s'']*W[w,s']*W[w,s'']
            # This is getting complex. Let me use a simpler alternating approach.
            
            # Simpler: fix two factors, solve for the third via least squares
            # Fix V and W, solve for U:
            # T_target[u,v,w] = sum_s U[u,s] * V[v,s] * W[w,s]
            # For each u: T_target[u,:,:] = sum_s U[u,s] * (V[:,s] * W[:,s])^T (as matrix)
            # This is: T_target[u,:,:] = sum_s U[u,s] * M_s where M_s = V[:,s] @ W[:,s]^T
            # So: T_target[u,:,:] = [M_0, M_1, ..., M_{r-1}] @ U[u,:]
            # This is a linear system in U[u,:] for each u
            
            # Build the linear system for U
            # For each u, we have: T_target[u,:,:] = sum_s U[u,s] * (V[:,s] * W[:,s])
            # Let's vectorize: T_target[u,v,w] = sum_s U[u,s] * V[v,s] * W[w,s]
            # This is equivalent to: for each (v,w), T_target[:,v,w] = sum_s U[:,s] * V[v,s] * W[w,s]
            # Let B[v,w,s] = V[v,s] * W[w,s], then T_target[:,v,w] = U @ B[v,w,:]
            
            # Solve for U using least squares:
            # T_target.reshape(nm, mp*pn) = U @ (B.reshape(r, mp*pn)).T
            B = np.einsum('vs,ws->vws', V, W)  # (mp, pn, r)
            B_flat = B.transpose(2, 0, 1).reshape(r, mp * pn)  # (r, mp*pn)
            T_flat = T_target.reshape(nm, mp * pn)  # (nm, mp*pn)
            
            # T_flat = U @ B_flat.T
            # U = T_flat @ pinv(B_flat.T)
            try:
                U_new = T_flat @ np.linalg.pinv(B_flat.T)
            except:
                U_new = U
            
            # Fix U and W, solve for V:
            B2 = np.einsum('us,ws->uws', U_new, W)  # (nm, pn, r)
            B2_flat = B2.transpose(2, 0, 1).reshape(r, nm * pn)  # (r, nm*pn)
            T_flat2 = T_target.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)
            try:
                V_new = T_flat2 @ np.linalg.pinv(B2_flat.T)
            except:
                V_new = V
            
            # Fix U and V, solve for W:
            B3 = np.einsum('us,vs->uvs', U_new, V_new)  # (nm, mp, r)
            B3_flat = B3.transpose(2, 0, 1).reshape(r, nm * mp)  # (r, nm*mp)
            T_flat3 = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)  # (pn, nm*mp)
            try:
                W_new = T_flat3 @ np.linalg.pinv(B3_flat.T)
            except:
                W_new = W
            
            U, V, W = U_new, V_new, W_new
        
        # Verify the best solution
        if best_solution is not None:
            U_best, V_best, W_best = best_solution
            if check_decomposition(U_best, V_best, W_best, tol=1e-6):
                return (U_best, V_best, W_best)
        
        return None

    def try_hybrid_approach(r, time_budget=25.0):
        """
        Hybrid: try simultaneous linear approach with multiple random restarts.
        """
        best_result = None
        best_rank = r + 1
        
        for attempt in range(15):
            if time.time() - start_time > time_limit - 8:
                break
            if time.time() - start_time > start_time + time_budget:
                break
            
            seed = 1000 + attempt * 17
            result = try_simultaneous_linear(r, max_iter=40, seed=seed)
            if result is not None:
                U, V, W = result
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
                    if R == 40:
                        return best_result
        
        return best_result

    # Try to find rank 40 first (theoretical lower bound)
    result = try_hybrid_approach(40, time_budget=30.0)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 41
    if time.time() - start_time < time_limit - 20:
        result = try_hybrid_approach(41, time_budget=25.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 42 (schoolbook) as fallback
    if time.time() - start_time < time_limit - 10:
        result = try_hybrid_approach(42, time_budget=15.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return get_schoolbook()
