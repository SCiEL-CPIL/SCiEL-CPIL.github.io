import numpy as np
import time
from scipy.optimize import minimize

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

    def try_rank(R, timeout=10.0, max_restarts=10):
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
            np.random.seed(42 + attempt)
            
            # Initialize U, V, W
            # Scale initialization to help convergence
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_comp - T
                return np.sum(diff**2)
            
            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-5:
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
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    # However, for 3x5x7, the rank is likely close to 105 or slightly less.
    # Let's try ranks from 104 down to 90.
    # We allocate a budget.
    deadline = time.time() + 110  # 110s safety margin
    
    # Try a few specific ranks that might be optimal
    # Often the rank is n*m*p - 1 or n*m*p - 2
    ranks_to_try = list(range(rank_school - 1, max(rank_school - 15, 90), -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Use up to 5 seconds per rank, or less if time is tight
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
