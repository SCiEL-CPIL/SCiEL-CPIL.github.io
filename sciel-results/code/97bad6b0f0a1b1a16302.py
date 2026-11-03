"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def compute_residual(U, V, W):
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R
    
    def objective_and_grad(params, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n * m, r)
        V = params[size_U:size_U + size_V].reshape(m * p, r)
        W = params[size_U + size_V:].reshape(p * n, r)
        
        R = compute_residual(U, V, W)
        loss = 0.5 * np.sum(R ** 2)
        
        grad_U = np.einsum('ijk,jc,kc->ic', R, V, W)
        grad_V = np.einsum('ijk,ic,kc->jc', R, U, W)
        grad_W = np.einsum('ijk,ic,jc->kc', R, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def check_and_round(U, V, W, tol=1e-6):
        """Check if entries are close to half-integers and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if close to half-integers
        if np.max(np.abs(U - U_r)) > tol:
            return False, None, None, None
        if np.max(np.abs(V - V_r)) > tol:
            return False, None, None, None
        if np.max(np.abs(W - W_r)) > tol:
            return False, None, None, None
        
        # Verify exactness with rounded values
        R = compute_residual(U_r, V_r, W_r)
        if np.max(np.abs(R)) < 1e-8:
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    def polish(U, V, W, r, max_iter=500, tol=1e-12):
        """High-precision polish using Gauss-Newton-like steps."""
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            R = compute_residual(U, V, W)
            if np.max(np.abs(R)) < tol:
                break
            
            # Compute gradients
            grad_U = np.einsum('ijk,jc,kc->ic', R, V, W)
            grad_V = np.einsum('ijk,ic,kc->jc', R, U, W)
            grad_W = np.einsum('ijk,ic,jc->kc', R, U, V)
            
            # Simple gradient descent with adaptive step
            step = 0.01
            for _ in range(10):
                U_new = U - step * grad_U
                V_new = V - step * grad_V
                W_new = W - step * grad_W
                R_new = compute_residual(U_new, V_new, W_new)
                if np.sum(R_new ** 2) < np.sum(R ** 2):
                    U, V, W = U_new, V_new, W_new
                    break
                step *= 0.5
            else:
                break
        
        return U, V, W
    
    # Schoolbook fallback
    def schoolbook():
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
    
    # Search ranks from low to high
    # Known lower bounds: 3x3x3 has rank 23, 3x3x5 likely >= 25
    min_rank = 23
    max_rank = n * m * p  # 45
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        num_restarts = max(5, int(remaining / 1.5))
        num_restarts = min(num_restarts, 50)
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(777 + restart * 137 + r * 31)
            
            # Try different initialization strategies
            init_type = restart % 3
            
            if init_type == 0:
                # Small random initialization
                params = np.random.randn(r * (n * m + m * p + p * n)) * 0.05
            elif init_type == 1:
                # Random with some structure
                params = np.random.randn(r * (n * m + m * p + p * n)) * 0.1
                # Set some entries to 0 or 1
                for idx in range(0, len(params), 5):
                    params[idx] = np.random.choice([0, 1, -1, 0.5, -0.5])
            else:
                # Sparse random initialization
                params = np.zeros(r * (n * m + m * p + p * n))
                nonzero = np.random.choice(len(params), size=int(len(params) * 0.3), replace=False)
                params[nonzero] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nonzero))
            
            try:
                # First pass: L-BFGS-B
                res = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r)[1],
                    method='L-BFGS-B',
                    options={'gtol': 1e-8, 'maxiter': 300, 'maxfun': 1000}
                )
                
                if res.fun < 1e-6:
                    size_U = n * m * r
                    size_V = m * p * r
                    U = res.x[:size_U].reshape(n * m, r)
                    V = res.x[size_U:size_U + size_V].reshape(m * p, r)
                    W = res.x[size_U + size_V:].reshape(p * n, r)
                    
                    # Polish phase
                    U, V, W = polish(U, V, W, r, max_iter=200)
                    
                    # Check for half-integer solution
                    is_exact, U_r, V_r, W_r = check_and_round(U, V, W)
                    if is_exact:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # If close but not exact, try more aggressive polishing
                    if np.max(np.abs(compute_residual(U, V, W))) < 1e-4:
                        U, V, W = polish(U, V, W, r, max_iter=500, tol=1e-14)
                        is_exact, U_r, V_r, W_r = check_and_round(U, V, W, tol=1e-5)
                        if is_exact:
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            except Exception:
                continue
        
        if found:
            break
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
