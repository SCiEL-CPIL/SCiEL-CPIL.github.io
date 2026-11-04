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
        num_attempts = 20 if r < 28 else 10
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Different seeds for variety
            np.random.seed(10000 + r * 1000 + attempt * 7)
            
            # Scale initialization to avoid large gradients
            scale = 1.0
            U0 = np.random.randn(n*m, r) * scale
            V0 = np.random.randn(m*p, r) * scale
            W0 = np.random.randn(p*n, r) * scale
            
            # Normalize columns of W to unit norm to help conditioning
            norms = np.linalg.norm(W0, axis=0, keepdims=True)
            norms[norms == 0] = 1
            W0 = W0 / norms
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # BFGS is often more robust for small problems
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.fun < 1e-6:
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
        
        # Try scaling factors 1, 2, 4, 8, 16
        for scale_factor in [1, 2, 4, 8, 16, 32]:
            U_s = U * scale_factor
            V_s = V * scale_factor
            W_s = W * scale_factor
            
            # Round to nearest integer (which corresponds to half-integers if scale=2, etc? 
            # Wait. If the entries are half-integers, 2*entries are integers.
            # So we check if scale_factor * entries are close to integers.
            # The problem says "integer/half-integer". 
            # So 2 * entries should be integers.
            # Let's check if 2 * U is close to integer, 2 * V is close to integer, 2 * W is close to integer.
            # If so, we return U, V, W as is (floats representing half-integers).
            # Or we return the floats. The evaluator accepts floats if they represent half-integers.
            
            # Check if 2 * U is close to integer
            U_check = U * 2
            V_check = V * 2
            W_check = W * 2
            
            if (np.allclose(U_check, np.rint(U_check), atol=1e-5) and
                np.allclose(V_check, np.rint(V_check), atol=1e-5) and
                np.allclose(W_check, np.rint(W_check), atol=1e-5)):
                
                # Verify the decomposition with rounded values
                U_r = np.rint(U_check) / 2.0
                V_r = np.rint(V_check) / 2.0
                W_r = np.rint(W_check) / 2.0
                
                T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_check, T_target, atol=1e-8):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    
            # Also try if U, V, W themselves are integers
            U_check_int = U
            V_check_int = V
            W_check_int = W
            
            if (np.allclose(U_check_int, np.rint(U_check_int), atol=1e-5) and
                np.allclose(V_check_int, np.rint(V_check_int), atol=1e-5) and
                np.allclose(W_check_int, np.rint(W_check_int), atol=1e-5)):
                
                U_r = np.rint(U_check_int)
                V_r = np.rint(V_check_int)
                W_r = np.rint(W_check_int)
                
                T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_check, T_target, atol=1e-8):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
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

    # Search ranks from low to high? Or high to low?
    # Low to high is better if we want the minimum rank.
    # But low ranks are harder to find.
    # Let's try 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35.
    for r in range(24, 36):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result

    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
