import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 280.0

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        evals_used = 0
        
        # 1. Global Search Phase: 10 random points
        # Use 10 evaluations to find a decent starting point
        num_init = 10
        X_init = np.random.uniform(-5, 5, size=(num_init, n))
        F_init = np.array([func(x) for x in X_init])
        evals_used += num_init
        
        best_idx = np.argmin(F_init)
        x = X_init[best_idx].copy()
        f_x = F_init[best_idx]
        best_x = x.copy()
        best_f = f_x
        
        # 2. DF-TBFGS Phase
        # Trust region radius
        delta = 1.0
        min_delta = 1e-4
        max_delta = 10.0
        
        # Hessian approximation
        H = np.eye(n)
        
        # Previous gradient
        # Calculate initial gradient
        eps = 1e-5
        g = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            # Note: func clips, so this is safe
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            evals_used += 2
            g[i] = (f_plus - f_minus) / (2 * eps)
            
        # BFGS history
        s_prev = None
        y_prev = None
        
        while evals_used < budget - 2 * n: # Need at least 2n for next gradient
            if time.time() > self.deadline:
                break
                
            # Solve Trust Region Subproblem
            # min 0.5 * p^T H p + g^T p
            # s.t. ||p|| <= delta
            
            # Simple approach: Newton step with line search, or solve QP
            # Since n=5, we can solve the QP using scipy or a simple iterative method.
            # scipy.optimize.minimize with method='L-BFGS-B' is available but might be overkill or slow if called repeatedly.
            # Let's use a simple steepest descent step scaled by delta if H is ill-conditioned, 
            # or solve the KKT system.
            
            # Actually, for small n, we can just use the Newton step if H is PD.
            # If H is not PD, use steepest descent.
            
            try:
                # Check if H is PD
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) > 1e-8:
                    p_newton = np.linalg.solve(H, -g)
                    norm_p = np.linalg.norm(p_newton)
                    if norm_p > delta:
                        p = p_newton * (delta / norm_p)
                    else:
                        p = p_newton
                else:
                    # Steepest descent
                    p = -g * (delta / np.linalg.norm(g))
            except:
                p = -g * (delta / np.linalg.norm(g))
                
            # Evaluate new point
            x_new = x + p
            # Clip to bounds? The trust region step might go out of bounds.
            # The function clips, so it's okay. But the gradient calculation assumes smoothness.
            # If we hit the boundary, the gradient estimate might be biased.
            # For now, let func handle clipping.
            
            f_new = func(x_new)
            evals_used += 1
            
            # Calculate actual improvement
            actual = f_x - f_new
            # Predicted improvement
            predicted = -g.T @ p - 0.5 * p.T @ H @ p
            
            rho = actual / predicted if predicted > 1e-12 else (1.0 if actual > 0 else 0.0)
            
            if rho > 0.25: # Accept
                # Update Hessian (BFGS)
                s = x_new - x
                # Calculate new gradient
                g_new = np.zeros(n)
                for i in range(n):
                    x_plus = x_new.copy()
                    x_minus = x_new.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals_used += 2
                    g_new[i] = (f_plus - f_minus) / (2 * eps)
                    
                y = g_new - g
                sy = s.T @ y
                if sy > 1e-12:
                    Hy = H @ y
                    Hyy = y.T @ Hy
                    # BFGS update
                    H = H - (H @ s @ s.T @ H) / (s.T @ H @ s) + (s @ s.T) / sy
                    # Symmetrize
                    H = (H + H.T) / 2
                    
                x = x_new
                f_x = f_new
                g = g_new
                s_prev = s
                y_prev = y
                
                if rho > 0.75:
                    delta = min(2 * delta, max_delta)
                # else keep delta same
                
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                    
            else: # Reject
                delta = max(0.5 * delta, min_delta)
                if delta <= min_delta:
                    # Stuck, break
                    break
                    
        return best_x, best_f
