"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hessian-based Newton method with finite difference estimation for Sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Explicit deadline guard: 300s limit, use 295s for safety
        start_time = time.time()
        deadline = start_time + 295
        
        best_x = None
        best_fx = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Phase 1: Initial exploration to find a reasonable starting point
        # Use a few random points and the origin to get a sense of the landscape
        # Sphere is convex, so the global minimum is unique.
        
        # Start at origin
        x = np.zeros(dim)
        fx = evaluate(x)
        
        # Try a few random points to see if we can find a better starting point
        # This is cheap and ensures we don't start in a bad spot if the shift is large
        for _ in range(10):
            x_rand = rng.uniform(-4, 4, dim)
            fx_rand = evaluate(x_rand)
            if fx_rand < fx:
                x = x_rand
                fx = fx_rand
                
        # Phase 2: Newton's method with finite difference Hessian and Gradient
        # We need to estimate H (dim x dim) and g (dim)
        # Finite difference step size
        # h should be small enough for approximation, large enough for numerical stability
        # For f(x) = (x-c)^2 + f0, derivatives are exact for any h if we use central differences?
        # f(x+h e_i) - f(x-h e_i) = 2 h (x-c)_i -> grad_i = (f(x+h)-f(x-h))/(2h)
        # f(x+h e_i) - 2f(x) + f(x-h e_i) = 2 h^2 (x-c)_i^2 -> H_ii = (f(x+h)-2f(x)+f(x-h))/h^2
        # H_ij = (f(x+h e_i + h e_j) - f(x+h e_i - h e_j) - f(x-h e_i + h e_j) + f(x-h e_i - h e_j)) / (4 h^2)
        
        # To save calls, we can estimate the Hessian using forward differences or a mix.
        # Central differences for Hessian are expensive: 2*dim + 2*dim*(dim-1)/2 = dim^2 + dim calls per iteration.
        # For dim=5, that's 30 calls per iteration.
        # We have ~480 calls left. 480 / 30 = 16 iterations.
        # Newton's method converges quadratically, so 5-10 iterations should be enough for high precision.
        
        # Let's use a simpler approach:
        # Estimate gradient using forward differences: dim calls.
        # Estimate Hessian using forward differences: dim*(dim+1)/2 calls.
        # Total per iteration: dim + dim*(dim+1)/2 = 5 + 15 = 20 calls.
        # 480 / 20 = 24 iterations.
        
        # Even better: Since it's a Sphere, the Hessian is constant and diagonal in the rotated frame.
        # But we don't know the rotation.
        
        # Let's stick to the standard finite difference Newton step.
        # h = 1e-4 is a good start.
        
        h = 1e-4
        
        for _ in range(20):  # Max 20 Newton iterations
            if calls >= self.budget or time.time() >= deadline:
                break
            
            # 1. Estimate Gradient g
            g = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                f_plus = evaluate(x_plus)
                
                # f(x) is already known as fx
                g[i] = (f_plus - fx) / h
                
            # 2. Estimate Hessian H (upper triangular)
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        # Second derivative
                        x_plus_plus = x.copy()
                        x_plus_plus[i] += 2*h
                        f_pp = evaluate(x_plus_plus)
                        # We already have f_plus (x+h) and fx (x)
                        H[i, i] = (f_pp - 2*f_plus + fx) / (h*h)
                    else:
                        # Mixed derivative
                        x_pp = x.copy()
                        x_pp[i] += h
                        x_pp[j] += h
                        f_pp = evaluate(x_pp)
                        
                        x_pm = x.copy()
                        x_pm[i] += h
                        x_pm[j] -= h
                        f_pm = evaluate(x_pm)
                        
                        x_mp = x.copy()
                        x_mp[i] -= h
                        x_mp[j] += h
                        f_mp = evaluate(x_mp)
                        
                        x_mm = x.copy()
                        x_mm[i] -= h
                        x_mm[j] -= h
                        f_mm = evaluate(x_mm)
                        
                        H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4*h*h)
                        H[j, i] = H[i, j]
                        
            # 3. Newton Step: x_new = x - H^-1 g
            try:
                # Add small regularization for stability if H is ill-conditioned
                # But for Sphere, H should be positive definite (lambda > 0)
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if H is singular
                step = g
                
            # Damping to avoid overshooting if the model is inaccurate
            # For quadratic functions, full Newton step is optimal.
            # But due to finite differences, it might not be perfect.
            # Let's try full step first.
            x_new = x - step
            
            # Check if step is too large or goes out of bounds significantly
            # If x_new is way out of bounds, damp it.
            # But func clips, so it's safe.
            
            fx_new = evaluate(x_new)
            
            if fx_new < fx:
                x = x_new
                fx = fx_new
                # Converged?
                if np.linalg.norm(step) < 1e-10:
                    break
            else:
                # Line search: try smaller steps
                alpha = 0.5
                improved = False
                while alpha > 1e-6:
                    x_trial = x - alpha * step
                    fx_trial = evaluate(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        break
                    alpha *= 0.5
                
                if not improved:
                    # Stuck? Break or try random restart?
                    # For Sphere, this shouldn't happen unless H is very wrong.
                    break

        # Phase 3: Polish with (1+1)-ES if we have budget left
        # This ensures we get machine precision if Newton stopped early
        sigma = 1e-6
        no_improve_count = 0
        
        while calls < self.budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
                no_improve_count = 0
            else:
                sigma *= 0.9
                no_improve_count += 1
                
            if sigma < 1e-15:
                break
            if no_improve_count > 50:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
