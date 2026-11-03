import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition(U, V, W):
        """
        Verifies if U, V, W exactly decompose T.
        Returns True if error is below tolerance.
        """
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_comp - T)**2)
        return error < 1e-9

    def try_rank(R, timeout=5.0, max_restarts=10):
        """
        Attempts to find a decomposition of rank R.
        Returns (U, V, W) as numpy arrays if successful, else None.
        """
        start_time = time.time()
        best_solution = None
        
        for attempt in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Different seeds for each attempt
            np.random.seed(42 + attempt * 13)
            
            # Initialize U, V, W
            # Scale initialization to help convergence
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_comp - T
                # Flatten the difference to match least_squares requirement
                return diff.flatten()
            
            # Use least_squares which is robust for this
            # method 'trf' is good for large problems
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=1000)
            except Exception:
                continue
            
            if res.cost < 1e-10:
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                if verify_decomposition(U_snap, V_snap, W_snap):
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        
        return best_solution

    # Search for lower rank
    # Schoolbook is 105.
    deadline = time.time() + 110
    
    # Try ranks from 104 down to 95
    # We prioritize lower ranks first as they give better scores
    ranks_to_try = list(range(104, 94, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Allocate time based on remaining budget
        # Give more time to higher ranks (closer to 105) as they are more likely to exist?
        # Or give equal time?
        # Let's give 5s per rank, or less if time is tight.
        t_limit = min(5.0, remaining / 2.0)
        
        result = try_rank(R, timeout=t_limit, max_restarts=5)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank = rank_school
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
