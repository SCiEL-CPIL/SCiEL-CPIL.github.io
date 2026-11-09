import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320.0  # Safety margin for 330s limit

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_used = 1
        
        # Parameters for gradient descent
        eps = 1e-4  # Perturbation for finite difference
        alpha = 0.1  # Initial step size
        min_alpha = 1e-6
        max_alpha = 10.0
        no_improve_count = 0
        max_no_improve = 10  # Restart if stuck for this many steps

        while calls_used < self.budget:
            # Time check
            if time.time() > self.deadline:
                break
            
            # Start a new local search from a random point or best point
            # Strategy: Alternate between random starts and polishing best
            if calls_used < 50:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Perturb the best point slightly to escape local minima/stagnation
                x = np.clip(x_best + rng.normal(0, 0.5, self.dim), -5, 5)
            
            fx = func(x)
            calls_used += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Gradient Descent Loop
            while calls_used + self.dim < self.budget: # Need at least dim evals for gradient
                if time.time() > self.deadline:
                    break
                
                # Compute gradient using one-sided finite differences
                grad = np.zeros(self.dim)
                for i in range(self.dim):
                    if calls_used >= self.budget:
                        break
                    x_pert = x.copy()
                    x_pert[i] = np.clip(x_pert[i] + eps, -5, 5)
                    fx_pert = func(x_pert)
                    calls_used += 1
                    grad[i] = (fx_pert - fx) / eps
                
                # Check for convergence/stagnation
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-8:
                    break
                
                # Propose new point
                x_new = np.clip(x - alpha * grad, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                
                if fx_new < fx:
                    x, fx = x_new, fx_new
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    # Increase step size if improving
                    alpha = min(alpha * 1.2, max_alpha)
                else:
                    # Decrease step size
                    alpha *= 0.5
                    no_improve_count += 1
                    if alpha < min_alpha:
                        break

            # If we got stuck, restart
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                alpha = 0.1

        return x_best, f_best
