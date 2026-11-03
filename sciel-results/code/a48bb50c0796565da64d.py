import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    n, m, p = 2, 5, 5
    
    # Precompute index mappings
    I, J, K = np.indices((n, m, p))
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I

    def als_step(U, V, W, max_iter=50, tol=1e-12):
        """
        Perform ALS iterations.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        
        for it in range(max_iter):
            # 1. Update U
            # For each i, j: solve for U[i*m+j, :]
            # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            U_new = np.copy(U)
            for i in range(n):
                for j in range(m):
                    # A is (p, R), b is (p,)
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve A x = b
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = sol
            
            # 2. Update V
            # For each j, k: solve for V[j*p+k, :]
            # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            V_new = np.copy(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = sol

            # 3. Update W
            # For each k, i: solve for W[k*n+i, :]
            # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            W_new = np.copy(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = sol
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Compute residual
            # This is expensive, maybe do it every few steps or at the end
            if it % 5 == 0 or it == max_iter - 1:
                U_gathered = U[U_idx, :]
                V_gathered = V[V_idx, :]
                W_gathered = W[W_idx, :]
                vals = np.sum(U_gathered * V_gathered * W_gathered, axis=3)
                res_norm = np.linalg.norm(vals - 1.0)
                if res_norm < tol:
                    break
                    
        return U, V, W, res_norm

    def try_rank(R, max_restarts=100):
        best_res_norm = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random init
            scale = 0.1
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            U, V, W, res_norm = als_step(U, V, W, max_iter=100)
            
            if res_norm < best_res_norm:
                best_res_norm = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if res_norm < 1e-10:
                return best_U, best_V, best_W, res_norm
                
        if best_U is not None and best_res_norm < 1e-8:
            return best_U, best_V, best_W, best_res_norm
        return None, None, None, best_res_norm

    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Try ranks from 38 upwards
    # ALS is fast, so we can try more restarts
    for R in range(38, 51):
        if time.time() > deadline - 10:
            break
            
        time_remaining = deadline - time.time() - 10
        if time_remaining <= 0:
            break
        
        # Estimate restarts. Each ALS run is ~ (n*m + m*p + p*n) * R * lstsq cost.
        # Very fast. Let's try 200 restarts if time permits.
        max_restarts = int(min(500, max(20, time_remaining * 100)))
        
        U_res, V_res, W_res, res_norm = try_rank(R, max_restarts=max_restarts)
        
        if U_res is not None:
            U_best, V_best, W_best = U_res, V_res, W_res
            best_rank = R
            break
            
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
