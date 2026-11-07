"""Rewritten program: Gradient probing for linear function optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer via gradient probing and vertex jump."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # Center point for probing
        x0 = np.zeros(d)
        f0 = func(x0)
        calls = 1
        best_x = x0.copy()
        best_f = f0
        
        # Probe positive axes to estimate gradient
        delta = 0.1
        grad = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = x0.copy()
            xi[i] += delta
            fi = func(xi)
            calls += 1
            grad[i] = (fi - f0) / delta
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Determine optimal vertex based on gradient signs
        x_vertex = np.zeros(d)
        for i in range(d):
            if grad[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
        
        # Evaluate at optimal vertex
        f_vertex = func(x_vertex)
        calls += 1
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
        
        # Use remaining budget for polish: try points near the vertex
        # and also try the interior minimum if gradient is small
        remaining = budget - calls
        if remaining > 0:
            # Try a few perturbations around best point
            rng = np.random.default_rng(42)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                # Small random perturbation
                x_try = best_x + 0.01 * rng.standard_normal(d)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
            
            # If we have more budget, do a finer local search
            if calls < budget:
                sigma = 0.01
                for _ in range(budget - calls):
                    if time.time() > deadline:
                        break
                    x_try = np.clip(best_x + sigma * rng.standard_normal(d), -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
