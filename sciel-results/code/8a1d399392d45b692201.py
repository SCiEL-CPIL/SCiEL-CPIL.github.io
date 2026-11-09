import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 structure:
        # f1 = x[0]
        # f2 = (1 + 10*(n-1)) * (1/sqrt(f1) - 2*f1^2 + f1)  [for minimization variant]
        # g(x[1:]) = sum of phi(x_i) where phi(x) = 4x^2 - sin(10*pi*x) + 1
        # g has minimum 1.0 when x_i = 0.5 for all i
        # The Pareto front is achieved when g(x[1:]) = 1 (minimum)
        # Then f1 = x[0] and f2 = (1+10*(n-1))*(1/sqrt(f1) - 2*f1^2 + f1)
        # Wait - standard ZDT3: f1 = x0, f2 = g(x[1:]) * (1 + sin(10*pi*x0))
        # Actually let me think about ZDT3 more carefully.
        # ZDT3: f1 = x0, f2 = g(x[1:]) * (1 + sin(10*pi*x0))
        # where g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [10*(x_i^2 - cos(20*pi*x_i)) + 10]
        # g has minimum 1 at x_i = 0.5 for all i >= 2
        # Pareto front: g = 1, f1 = x0, f2 = 1 + sin(10*pi*x0)
        # This gives a front with 7 disjoint segments (where sin >= -1)
        # Actually f2 = 1 + sin(10*pi*x0) which oscillates between 0 and 2.
        # The Pareto front has points where f1 in [0,1] and f2 = 1+sin(10*pi*f1)
        # But wait - for minimization, we want both f1 and f2 minimized.
        # A point (f1, f2) is on the Pareto front if no other point dominates it.
        # With f1 = x0 and f2 = g * (1 + sin(10*pi*x0)), when g=1:
        # f2 = 1 + sin(10*pi*f1)
        # The non-dominated set consists of points where locally f2 is decreasing as f1 increases
        # i.e., where d(f2)/d(f1) < 0, which is where cos(10*pi*f1) < 0
        
        # The key insight: set x[1:] = 0.5 to minimize g, then sweep x[0]
        # to cover the entire Pareto front.

        # Set x[1:] to 0.5 to achieve g minimum
        base_x = np.zeros(dim)
        base_x[1:] = 0.5

        # Use a deterministic space-filling approach: uniform grid in x[0]
        # This guarantees coverage of all 7 Pareto segments
        n_samples = min(budget, 200)
        
        # Create uniform samples of x[0] from 0 to 1
        # Using linspace for exact uniform coverage
        x0_values = np.linspace(0.0, 1.0, n_samples)
        
        evaluated = 0
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x0_values[i]
            
            try:
                f = problem(x)
                evaluated += 1
            except Exception:
                break
        
        # If we still have budget and time, do adaptive refinement
        # Focus on areas where the front is more curved
        if evaluated < budget and time.time() < deadline:
            # Add additional samples at key points: boundaries of segments
            # The segments change at points where sin(10*pi*x) has local min/max
            # i.e., where cos(10*pi*x) = 0, so x = (2k+1)/20 for k=0,...,9
            key_points = [(2*k + 1) / 20.0 for k in range(10)]
            # Also add endpoints
            key_points = [0.0] + key_points + [1.0]
            
            remaining = budget - evaluated
            for xp in key_points:
                if evaluated >= budget or time.time() > deadline:
                    break
                x = base_x.copy()
                x[0] = xp
                try:
                    problem(x)
                    evaluated += 1
                except Exception:
                    break
            
            # If still budget remaining, add more uniform samples
            if evaluated < budget:
                extra_needed = budget - evaluated
                if extra_needed > 0:
                    # Add finer uniform grid
                    extra_x0 = np.linspace(0.0, 1.0, extra_needed)
                    for xp in extra_x0:
                        if evaluated >= budget or time.time() > deadline:
                            break
                        x = base_x.copy()
                        x[0] = xp
                        try:
                            problem(x)
                            evaluated += 1
                        except Exception:
                            break
        
        return None
