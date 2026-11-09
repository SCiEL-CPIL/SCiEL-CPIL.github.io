"""Rewritten optimizer using finite-difference gradient descent for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Finite-difference gradient descent with adaptive step size for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Start from a reasonable initial point
        x = rng.uniform(-3, 3, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Finite-difference step size
        h = 1e-4
        # Learning rate / step size
        lr = 0.5
        lr_min = 1e-6
        lr_max = 2.0
        patience = 0
        max_patience = 30
        
        while calls < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Compute finite-difference gradient: 2*dim calls
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls += 1
                
                if calls >= self.budget:
                    break
                x_minus = x.copy()
                x_minus[i] -= h
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = func(x_minus)
                calls += 1
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            if calls >= self.budget:
                break
            
            # Check if gradient is negligible
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                # Try random restart
                x = rng.uniform(-4, 4, self.dim)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                continue
            
            # Normalize gradient and take step
            direction = -grad / grad_norm
            step = lr * direction
            
            # Propose new point
            x_new = np.clip(x + step, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                # Accept
                x = x_new
                fx = f_new
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                patience = 0
                # Increase learning rate cautiously
                lr = min(lr * 1.1, lr_max)
            else:
                # Reject
                patience += 1
                lr *= 0.8
                if lr < lr_min:
                    lr = lr_min
                    # If stuck, try a larger random perturbation
                    if patience > max_patience:
                        x = x + rng.normal(0, 1.0, self.dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls += 1
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                        patience = 0
                        lr = 0.3
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
