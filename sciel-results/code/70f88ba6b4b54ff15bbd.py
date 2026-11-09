import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for rotated Rosenbrock (f9).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')

        # Custom exception to stop optimization early
        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            # Check time limit with safety margin
            if time.time() - self.start_time > 290:
                raise BudgetExceeded()
            
            # Check call budget
            if self.eval_count >= self.budget:
                raise BudgetExceeded()

            # Clip input to box [-5, 5]
            x = np.clip(x, -5, 5)
            
            self.eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10  # Large penalty for errors

            # Update global best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            return fx

        # Generate diverse starting points
        # The minimum is in [-4, 4]^5.
        # We use a mix of the center and random points.
        rng = np.random.default_rng(12345)
        
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        
        # Random points in [-4, 4]
        num_random_starts = 15
        for _ in range(num_random_starts):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # Add a few points near the boundaries of the likely minimum region
        for _ in range(5):
             starts.append(rng.choice([-4, 4], size=self.dim))

        bounds = [(-5, 5)] * self.dim

        for x0 in starts:
            # Check if we have enough budget left for a meaningful run
            # L-BFGS-B typically needs ~20-50 calls to converge in 5D
            if self.budget - self.eval_count < 10:
                break
            
            if time.time() - self.start_time > 290:
                break

            try:
                # Run L-BFGS-B
                # maxiter is high, but we rely on budget/time checks
                # ftol/gtol are tight to ensure convergence
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                # The wrapper has already updated self.best_x and self.best_f
            except BudgetExceeded:
                break
            except Exception:
                # If minimize fails for any reason, continue to next start
                continue

        return self.best_x, self.best_f
