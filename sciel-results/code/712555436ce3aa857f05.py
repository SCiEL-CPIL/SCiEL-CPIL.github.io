import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute target tensor
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
            
        # Objective function
        def objective_and_grad(params):
            if time.time() > deadline:
                return np.inf, np.zeros_like(params)
            
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - TARGET_TENSOR
            obj = np.sum(diff**2)
            
            # Gradients
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad
        
        # Try several random initializations
        # Increase attempts for lower ranks as they are harder
        num_attempts = 50 if r <= 26 else 30
        found = None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Use a deterministic seed based on rank and attempt for reproducibility in debugging, 
            # but random in production would be better. Here we use a mix.
            np.random.seed(10000 + r * 1000 + attempt * 7)
            
            # Strategy 1: Sparse Integer Initialization
            # Many tensor decompositions for small matrices have sparse factor matrices.
            U0 = np.zeros((n*m, r))
            V0 = np.zeros((m*p, r))
            W0 = np.zeros((p*n, r))
            
            # Randomly place non-zero entries
            density = 0.4
            for _ in range(int(n*m*r*density)):
                i = np.random.randint(n*m)
                j = np.random.randint(r)
                U0[i, j] = np.random.choice([-1, 1])
                
            for _ in range(int(m*p*r*density)):
                i = np.random.randint(m*p)
                j = np.random.randint(r)
                V0[i, j] = np.random.choice([-1, 1])
                
            for _ in range(int(p*n*r*density)):
                i = np.random.randint(p*n)
                j = np.random.randint(r)
                W0[i, j] = np.random.choice([-1, 1])
            
            # Sometimes add small noise to help escape local minima
            if attempt % 5 == 0:
                U0 += np.random.randn(n*m, r) * 0.1
                V0 += np.random.randn(m*p, r) * 0.1
                W0 += np.random.randn(p*n, r) * 0.1

            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # L-BFGS-B is more efficient for this number of parameters
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50}
                )
                
                if res.fun < 1e-4:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    rational_solution = recover_rational(U, V, W, r, n, m, p, TARGET_TENSOR)
                    if rational_solution is not None:
                        return rational_solution
                        
            except Exception:
                continue
        
        return None

    def recover_rational(U, V, W, r, n, m, p, T_target):
        """Try to recover integer/half-integer values."""
        if time.time() > deadline:
            return None
        
        # The problem states entries are integer or half-integer.
        # This means 2 * entries are integers.
        
        U_check = U * 2
        V_check = V * 2
        W_check = W * 2
        
        # Check if they are close to integers
        U_int = np.rint(U_check)
        V_int = np.rint(V_check)
        W_int = np.rint(W_check)
        
        # Calculate error of snapping
        err_U = np.max(np.abs(U_check - U_int))
        err_V = np.max(np.abs(V_check - V_int))
        err_W = np.max(np.abs(W_check - W_int))
        
        if err_U < 1e-3 and err_V < 1e-3 and err_W < 1e-3:
            U_r = U_int / 2.0
            V_r = V_int / 2.0
            W_r = W_int / 2.0
            
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # If snapping fails, try to polish with a few Newton steps constrained to half-integers?
        # Or just accept if it's extremely close and force round?
        # Let's try a local search around the rounded values
        if err_U < 0.1 and err_V < 0.1 and err_W < 0.1:
            # Try perturbing the rounded values
            for _ in range(10):
                U_p = U_int + np.random.choice([-1, 0, 1], size=(n*m, r)) * 0.01
                V_p = V_int + np.random.choice([-1, 0, 1], size=(m*p, r)) * 0.01
                W_p = W_int + np.random.choice([-1, 0, 1], size=(p*n, r)) * 0.01
                
                # Re-snap
                U_p = np.rint(U_p) / 2.0
                V_p = np.rint(V_p) / 2.0
                W_p = np.rint(W_p) / 2.0
                
                T_check = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                if np.allclose(T_check, T_target, atol=1e-8):
                    return {
                        "U": U_p.tolist(),
                        "V": V_p.tolist(),
                        "W": W_p.tolist()
                    }
        
        return None

    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Search ranks from low to high
    # Known upper bound for 3x3x4 is likely < 36.
    # Literature suggests ranks around 24-28.
    for r in range(24, 36):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result

    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
