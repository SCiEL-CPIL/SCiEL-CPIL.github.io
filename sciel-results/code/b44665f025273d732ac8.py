"""Rewritten program using Model-and-Optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using linear model estimation followed by (1+1)-ES refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Hard kill limit is 330s. We set a safety margin.
        deadline = start_time + 290.0

        rng = np.random.default_rng(123)
        dim = self.dim

        # Phase 1: Estimate Linear Model Parameters
        # We assume f(x) ≈ w^T x + b
        # We estimate w using central differences at x=0
        x0 = np.zeros(dim)
        
        # Evaluate center
        f0 = func(x0)
        calls = 1
        x_best = x0.copy()
        f_best = f0
        
        gradients = np.zeros(dim)
        
        # Use multiple step sizes for robustness
        h_values = [0.1, 0.5, 1.0]
        grad_estimates = []
        
        for h in h_values:
            if time.time() > deadline or calls >= self.budget:
                break
            grad_h = np.zeros(dim)
            for i in range(dim):
                if time.time() > deadline or calls >= self.budget:
                    break
                xp = x0.copy()
                xm = x0.copy()
                xp[i] += h
                xm[i] -= h
                
                fxp = func(xp)
                calls += 1
                if fxp < f_best:
                    f_best = fxp
                    x_best = xp.copy()
                
                if time.time() > deadline or calls >= self.budget:
                    break
                    
                fxm = func(xm)
                calls += 1
                if fxm < f_best:
                    f_best = fxm
                    x_best = xm.copy()
                    
                grad_h[i] = (fxp - fxm) / (2.0 * h)
            grad_estimates.append(grad_h)
        
        # Average gradient estimates for robustness
        if grad_estimates:
            gradients = np.mean(grad_estimates, axis=0)
        else:
            # Fallback: simple forward differences
            h = 0.1
            for i in range(dim):
                if time.time() > deadline or calls >= self.budget:
                    break
                xp = x0.copy()
                xp[i] += h
                fxp = func(xp)
                calls += 1
                if fxp < f_best:
                    f_best = fxp
                    x_best = xp.copy()
                gradients[i] = (fxp - f0) / h

        # Phase 2: Predict Optimum from Linear Model
        # For a linear function, the minimum on the box [-5,5]^dim is at the vertex
        # opposite to the gradient direction.
        x_pred = np.zeros(dim)
        for i in range(dim):
            if gradients[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
                
        if time.time() <= deadline and calls < self.budget:
            f_pred = func(x_pred)
            calls += 1
            if f_pred < f_best:
                f_best = f_pred
                x_best = x_pred.copy()
        else:
            x_pred = x_best.copy()

        # Phase 3: Refinement with (1+1)-ES
        # Use remaining budget to polish the solution.
        # This handles any non-linearity or estimation errors.
        x = x_best.copy()
        fx = f_best
        sigma = 0.5
        
        remaining = self.budget - calls
        
        # Track best in this phase
        x_phase_best = x.copy()
        f_phase_best = fx
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                if fx < f_phase_best:
                    f_phase_best = fx
                    x_phase_best = x.copy()
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            if sigma < 1e-8:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
