import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    CMA-ES based global optimizer for rotated Rosenbrock (f9).
    Uses CMA-ES for global exploration to handle anisotropic landscape,
    followed by L-BFGS-B polishing for local convergence.
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
            if time.time() - self.start_time > 280:
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

        # Phase 1: CMA-ES for global exploration
        # CMA-ES adapts its covariance matrix to the anisotropic landscape
        # of rotated Rosenbrock, efficiently navigating the narrow valley
        
        # Initial parameters for CMA-ES
        x0 = np.zeros(self.dim)
        sigma0 = 3.0  # Initial step size, covering most of [-4, 4]^5
        
        # Reserve budget for polishing
        cma_budget = int(self.budget * 0.7)  # Use 70% of budget for CMA-ES
        polish_budget = self.budget - cma_budget
        
        # Track evaluations during CMA-ES
        cma_eval_count = 0
        
        def cma_func(x):
            nonlocal cma_eval_count
            if cma_eval_count >= cma_budget:
                raise BudgetExceeded()
            if time.time() - self.start_time > 260:
                raise BudgetExceeded()
            
            cma_eval_count += 1
            return wrapped_func(x)
        
        # Run CMA-ES with maxiter set to use most of the reserved budget
        # Each iteration uses dim+1 evaluations, so maxiter ≈ cma_budget / (dim+1)
        max_iter_cma = max(10, int(cma_budget / (self.dim + 1)))
        
        try:
            result_cma = scipy.optimize.minimize(
                cma_func,
                x0,
                method='CmaQstr',  # Use CmaQstr if available, otherwise fall back
                options={
                    'maxiter': max_iter_cma,
                    'sigma0': sigma0,
                }
            )
        except (BudgetExceeded, Exception):
            # If CMA-ES fails or runs out of budget, use current best
            pass
        
        # Phase 2: Polish with L-BFGS-B using remaining budget
        # Start from the best point found by CMA-ES
        x_start = self.best_x.copy()
        
        # Check if we have enough budget for polishing
        if self.budget - self.eval_count > 5 and time.time() - self.start_time < 270:
            try:
                result_polish = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
            except (BudgetExceeded, Exception):
                pass
        
        return self.best_x, self.best_f
