"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Newton-style direct minimizer estimation using quadratic structure
        # Sphere is f(x) = ||Ax - b||^2 + c for some hidden linear transform.
        # We estimate the minimizer by finite-difference gradient + Hessian.
        
        # Start with a few strategic points to get an initial estimate
        # Use center and axis points to estimate gradient
        center = np.zeros(dim)
        fc = func(center)
        calls += 1
        if fc < best_f:
            best_f = fc
            best_x = center.copy()
        
        # Estimate gradient at center using finite differences
        h = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            xp = center.copy()
            xp[i] += h
            xp = clip(xp)
            fplus = func(xp)
            calls += 1
            if fplus < best_f:
                best_f = fplus
                best_x = xp.copy()
            
            xm = center.copy()
            xm[i] -= h
            xm = clip(xm)
            fminus = func(xm)
            calls += 1
            if fminus < best_f:
                best_f = fminus
                best_x = xm.copy()
            
            grad[i] = (fplus - fminus) / (2 * h)
        
        # For Sphere, f(x) = (x-x*)^T A (x-x*) + f_opt where A is positive definite
        # Gradient at x: 2*A*(x-x*)
        # If we can estimate A and gradient, we can solve for x*
        # But A is unknown. Instead, use a simpler approach:
        # Take Newton step assuming identity Hessian first, then refine
        
        # Actually, let's use a more robust approach:
        # Multi-scale (1+1)-ES with adaptive sigma, starting from multiple scales
        
        # Phase 2: Multi-start (1+1)-ES with progressively smaller initial sigmas
        # This leverages the convexity of Sphere
        
        # First, try to get closer to the minimizer using gradient info
        # If gradient is small, we're close; otherwise take a step
        
        x = center.copy()
        fx = fc
        
        # If gradient is significant, take a step in negative gradient direction
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6 and calls < budget:
            # Take a step; for sphere, optimal step size depends on curvature
            # Use a conservative step
            step_size = min(1.0, 10.0 / grad_norm)
            x_new = clip(x - step_size * grad)
            fx_new = func(x_new)
            calls += 1
            if fx_new < fx:
                x, fx = x_new, fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Phase 3: (1+1)-ES with adaptive sigma from current best
        # Use multiple restarts with different initial sigmas
        
        sigmas_to_try = [2.0, 1.0, 0.5, 0.2, 0.1]
        rng = np.random.default_rng(42)
        
        for sigma_init in sigmas_to_try:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Start from current best
            x_local = best_x.copy()
            fx_local = best_f
            sigma = sigma_init
            
            # Run (1+1)-ES for a limited number of iterations
            max_iter_local = min(50, budget - calls)
            for _ in range(max_iter_local):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = clip(x_local + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    sigma *= 1.2
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma *= 0.8
                
                if sigma < 1e-8:
                    break
            
            # If we made progress, continue with smaller sigma
            if fx_local < best_f + 1e-10:
                continue
            else:
                # No progress, try next sigma
                break
        
        # Phase 4: Final polish with very small steps
        if calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            fx_local = best_f
            sigma = 0.01
            
            max_iter_polish = min(50, budget - calls)
            for _ in range(max_iter_polish):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = clip(x_local + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    sigma *= 1.1
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma *= 0.7
                
                if sigma < 1e-10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
