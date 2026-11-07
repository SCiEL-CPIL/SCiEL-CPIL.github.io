import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        # We will use a model-based approach
        # Estimate gradient and Hessian locally
        
        calls_used = 1
        best_x = x
        best_f = fx
        
        # Helper to estimate gradient and Hessian
        # Using central differences
        # h = 1e-4
        
        while calls_used < self.budget:
            # Check if we have enough budget for a full model update
            # Grad: 2*d, Hessian: d*(d+1)/2 * 2? 
            # Central diff for grad: 2d
            # Central diff for Hessian: d*(d+1) roughly (using symmetry)
            # Let's use forward differences to save calls if budget is low, 
            # but we have plenty.
            
            h = 1e-4
            
            # Estimate Gradient
            if calls_used + 2 * self.dim > self.budget:
                break
            
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                # Clip to box? The function clips internally? 
                # "func(x) clips a finite length-5 vector to the box"
                # So we can pass out-of-bounds, but it's better to stay in bounds.
                # However, central diff at boundary is problematic.
                # If x[i] is close to 5, xm is fine, xp is clipped to 5.
                # This makes gradient estimate biased at boundary.
                # But min is in [-4, 4], so we should be away from boundary.
                
                f_p = func(xp)
                f_m = func(xm)
                calls_used += 2
                grad[i] = (f_p - f_m) / (2 * h)
                
            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                break
                
            # Estimate Hessian
            # Only if we have budget
            hess_calls_needed = self.dim * (self.dim + 1) # Approx for central diff of 2nd deriv
            # Actually, for symmetric Hessian, we need d(d+1)/2 entries.
            # Each entry needs 2 evals (central).
            # Total d(d+1) evals.
            
            if calls_used + self.dim * (self.dim + 1) > self.budget:
                # Not enough for Hessian, do gradient descent step
                # Simple line search
                alpha = 0.1
                # Try a few step sizes
                best_step_f = fx
                best_step_x = x
                for factor in [0.5, 1.0, 2.0]:
                    a = alpha * factor
                    xn = x - a * grad
                    fn = func(xn)
                    calls_used += 1
                    if fn < best_step_f:
                        best_step_f = fn
                        best_step_x = xn
                x = best_step_x
                fx = best_step_f
                if fx < best_f:
                    best_f = fx
                    best_x = x
                continue
                
            # Compute Hessian
            hess = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(i, self.dim):
                    xpp = x.copy()
                    xpm = x.copy()
                    xmp = x.copy()
                    xmm = x.copy()
                    
                    xpp[i] += h; xpp[j] += h
                    xpm[i] += h; xpm[j] -= h
                    xmp[i] -= h; xmp[j] += h
                    xmm[i] -= h; xmm[j] -= h
                    
                    f_pp = func(xpp)
                    f_pm = func(xpm)
                    f_mp = func(xmp)
                    f_mm = func(xmm)
                    
                    calls_used += 4
                    hess_val = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    hess[i, j] = hess_val
                    hess[j, i] = hess_val
                    
            # Solve H * d = -grad
            # Add regularization
            lambda_reg = 1e-4
            H_reg = hess + lambda_reg * np.eye(self.dim)
            
            try:
                step = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                step = -grad * 0.1 # Fallback
                
            # Line search
            alpha = 1.0
            best_step_f = fx
            best_step_x = x
            for _ in range(10):
                xn = x + alpha * step
                fn = func(xn)
                calls_used += 1
                if fn < best_step_f:
                    best_step_f = fn
                    best_step_x = xn
                else:
                    alpha *= 0.5
                    if alpha < 1e-8:
                        break
                        
            x = best_step_x
            fx = best_step_f
            
            if fx < best_f:
                best_f = fx
                best_x = x
                
            # Check for stagnation
            if abs(fx - best_f) < 1e-10:
                break
                
        return best_x, best_f
