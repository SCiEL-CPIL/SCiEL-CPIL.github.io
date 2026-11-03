import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    target = np.ones((n, m, p))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return (U.reshape(n*m, rank).tolist(), 
                V.reshape(m*p, rank).tolist(), 
                W.reshape(p*n, rank).tolist())

    U_fb, V_fb, W_fb = schoolbook()
    
    def verify_exact(U, V, W, R):
        # Check half-integer property
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-8):
                return False
        # Check exact reconstruction
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        if not np.allclose(S, target, atol=1e-8):
            return False
        return True

    def als_step(U, V, W, R, max_iters=100, tol=1e-12):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = V[j, :, :] * W[:, i, :]  # (p, R)
                    b = np.ones(p)
                    u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i, j, :] = u
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    A = U[:, j, :] * W[k, :, :]  # (n, R)
                    b = np.ones(n)
                    v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j, k, :] = v
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    A = U[i, :, :] * V[:, k, :]  # (m, R)
                    b = np.ones(m)
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k, i, :] = w
            
            # Compute cost
            S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            cost = np.sum((S - target)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
            
        return U, V, W, cost

    # Candidate ranks to try
    candidate_ranks = [38, 39, 40, 41, 42, 44, 46, 48]
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        max_restarts = 30 if R <= 40 else 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            
            # Small random initialization
            scale = 0.1
            U_init = np.random.randn(n, m, R) * scale
            V_init = np.random.randn(m, p, R) * scale
            W_init = np.random.randn(p, n, R) * scale
            
            U_sol, V_sol, W_sol, cost = als_step(U_init, V_init, W_init, R)
            
            # Check if cost is close to 0
            if cost < 1e-6:
                # Round to half-integers
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                if verify_exact(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.reshape(n*m, R).tolist(),
                        "V": V_round.reshape(m*p, R).tolist(),
                        "W": W_round.reshape(p*n, R).tolist()
                    }
                
                # If rounding failed, try a polish step?
                # If cost is very low, the solution is close to a valid one.
                # It's possible that the exact solution requires specific rational/half-integer values
                # that the optimizer didn't hit exactly due to floating point.
                # We can try to "snap" the values to the nearest half-integers and check again.
                # If that fails, we might need to refine the solution further.
                # For now, we just try the next restart.

    # Fallback
    return {"U": U_fb, "V": V_fb, "W": W_fb}
