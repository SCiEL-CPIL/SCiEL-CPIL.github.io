import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Searches for a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Build the target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook fallback
    def make_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = make_schoolbook()
    best_rank = n * m * p

    def try_rank(r, seeds=10):
        if time.time() > deadline - 20:
            return None
            
        # Objective function
        def objective(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T - Rec
            return np.sum(Res * Res)

        # Gradient function
        def gradient(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T - Rec
            Grad_U = np.einsum('ijl,jc,lc->ic', Res, V, W)
            Grad_V = np.einsum('ijl,ic,lc->jc', Res, U, W)
            Grad_W = np.einsum('ijl,ic,jc->lc', Res, U, V)
            return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])

        for seed in range(seeds):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(10000 + r * 100 + seed)
            # Initialize with small random values to avoid zero solution
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5
            
            try:
                # Use L-BFGS-B for large scale optimization
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
                
                if res.fun < 1e-6:
                    x_opt = res.x
                    U = x_opt[:n*m*r].reshape(n*m, r)
                    V = x_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = x_opt[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness with snapped values
                    Rec_snap = np.einsum('ik,jk,lk->ijl', U_snap, V_snap, W_snap)
                    err_snap = np.linalg.norm(T - Rec_snap)
                    
                    if err_snap < 1e-6:
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
                
        return None

    # Search for lower ranks
    # Based on cell findings, rank 46 is achievable.
    # We try 49 down to 40 to find the best possible within time.
    for r in range(49, 39, -1):
        res = try_rank(r)
        if res is not None:
            best_result = res
            best_rank = r
            # If we found a significantly better rank, we can break or continue.
            # Given time constraints, we might want to try a few more if fast,
            # but L-BFGS is slow. We'll break if we find 46 or lower to save time?
            # No, we want the MINIMAL rank. So we should continue down if possible.
            # However, if we find 46, trying 45 might take too long.
            # Let's add a time check.
            if time.time() > deadline - 30:
                break

    return best_result
