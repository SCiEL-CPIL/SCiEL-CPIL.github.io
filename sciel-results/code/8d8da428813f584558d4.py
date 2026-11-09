"""Derivative-Free Trust-Region optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """
    Derivative-Free Trust-Region (DF-T) optimizer.
    Strategy:
      - Maintain a local quadratic model within a trust region.
      - Use a small set of points to fit the model (least squares).
      - Update the trust region based on the ratio of actual to predicted improvement.
      - Restart with a new trust region centered at the best point if stuck.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        dim = self.dim
        n_evals = 0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # Initial random sampling to find a good starting point
        for _ in range(10):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Trust region optimization
        # We will maintain a set of points in the trust region
        # and fit a quadratic model to them.
        
        # Initial trust region radius
        delta = 1.0
        
        # Number of points to maintain in the model
        n_model_points = 12  # Enough to fit a quadratic in 5D (1 + 5 + 10 = 16 terms, but we use a simpler model)
        
        # We'll use a linear model first, then maybe upgrade
        # For simplicity, let's use a quadratic model with a fixed set of points
        
        # Points in the trust region
        points = [best_x.copy()]
        fvals = [best_f]
        
        # Generate initial points for the model
        for i in range(n_model_points - 1):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
            # Generate points in the trust region
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            x_new = best_x + delta * direction
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            n_evals += 1
            points.append(x_new)
            fvals.append(f_new)
        
        # Main trust region loop
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Fit a quadratic model to the points
            # We'll use a simple least squares fit
            X = np.array(points)
            f = np.array(fvals)
            
            # Center the points at best_x
            X_centered = X - best_x
            
            # Build the design matrix for a quadratic model
            # f(x) ~ c0 + c1^T (x - x0) + (x - x0)^T C (x - x0)
            # We'll use a simplified approach: fit a linear model first
            # For now, let's use a linear model
            A = np.hstack([X_centered, np.ones((len(X), 1))])
            
            # Solve the least squares problem
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, f, rcond=None)
            except:
                # If the fit fails, restart
                break
            
            # The linear model is: f(x) ~ coeffs[0:dim] @ (x - x0) + coeffs[dim]
            # The predicted improvement direction is -coeffs[0:dim]
            grad = coeffs[0:dim]
            
            # If the gradient is very small, we're at a local minimum
            if np.linalg.norm(grad) < 1e-6:
                # Shrink the trust region
                delta *= 0.5
                if delta < 1e-4:
                    # Restart
                    delta = 1.0
                    points = [best_x.copy()]
                    fvals = [best_f]
                    for i in range(n_model_points - 1):
                        if n_evals >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction)
                        x_new = best_x + delta * direction
                        x_new = np.clip(x_new, -5, 5)
                        f_new = func(x_new)
                        n_evals += 1
                        points.append(x_new)
                        fvals.append(f_new)
                    continue
            
            # Normalize the gradient
            grad_norm = np.linalg.norm(grad)
            grad_dir = -grad / grad_norm
            
            # Take a step in the trust region
            step_size = min(delta, 1.0 / grad_norm)
            x_trial = best_x + step_size * grad_dir
            x_trial = np.clip(x_trial, -5, 5)
            
            # Evaluate the trial point
            f_trial = func(x_trial)
            n_evals += 1
            
            # Compute the actual and predicted improvement
            actual_improvement = best_f - f_trial
            predicted_improvement = grad_norm * step_size
            
            # Update the trust region
            if actual_improvement > 0.25 * predicted_improvement:
                # Good step, expand the trust region
                delta = min(2.0 * delta, 5.0)
                best_x = x_trial.copy()
                best_f = f_trial
            elif actual_improvement > 0.1 * predicted_improvement:
                # Okay step, keep the trust region
                best_x = x_trial.copy()
                best_f = f_trial
            else:
                # Bad step, shrink the trust region
                delta *= 0.5
            
            # Update the model points
            # Remove the point farthest from the center
            if len(points) >= n_model_points:
                distances = np.linalg.norm(np.array(points) - best_x, axis=1)
                farthest_idx = np.argmax(distances)
                points.pop(farthest_idx)
                fvals.pop(farthest_idx)
            
            # Add the new point
            points.append(x_trial.copy())
            fvals.append(f_trial)
            
            # If we're stuck, restart
            if delta < 1e-3:
                delta = 1.0
                points = [best_x.copy()]
                fvals = [best_f]
                for i in range(n_model_points - 1):
                    if n_evals >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    x_new = best_x + delta * direction
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    n_evals += 1
                    points.append(x_new)
                    fvals.append(f_new)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
