"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

class Optimizer:
    """Two-phase surrogate-assisted optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise ValueError("Timeout")
            if calls >= total_budget:
                raise ValueError("Budget exceeded")
            
            # Ensure x is a valid vector
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                raise ValueError("Invalid shape")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite values")
                
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
                
            return f

        # Phase 1: Exploration for Surrogate
        # Use 50 evaluations to build a surrogate model
        n_explore = min(50, total_budget)
        X_samples = []
        Y_samples = []
        
        # Generate diverse points: mix of random and structured
        # Simple random uniform
        rng = np.random.default_rng(42)
        for _ in range(n_explore):
            x = rng.uniform(-5, 5, size=n)
            f = safe_func(x)
            X_samples.append(x)
            Y_samples.append(f)
            
        X_samples = np.array(X_samples)
        Y_samples = np.array(Y_samples)

        # Build Surrogate (RBF Interpolator)
        rbf = None
        try:
            # RBFInterpolator can be sensitive to scaling, but [-5,5] is fine
            # kernel='multiquadric' is often robust
            rbf = RBFInterpolator(X_samples, Y_samples, kernel='multiquadric', smoothing=1e-3)
        except Exception:
            pass

        # Phase 2: Find a good starting point using the surrogate
        x_start = best_x.copy()
        if rbf is not None:
            # Use a few random restarts to minimize the surrogate
            best_surrogate_f = np.inf
            for _ in range(5):
                x0 = rng.uniform(-5, 5, size=n)
                try:
                    res = minimize(lambda x: rbf(x.reshape(1, -1))[0], x0, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.success and res.fun < best_surrogate_f:
                        best_surrogate_f = res.fun
                        x_start = res.x
                except Exception:
                    pass
                    
        # Phase 3: Local Optimization on the True Function
        # Use LBFGS-B with the remaining budget
        remaining_budget = total_budget - calls
        if remaining_budget > 0:
            try:
                # LBFGS-B uses line searches. We limit maxiter to ensure we don't blow the budget.
                # A conservative maxiter is 50-100.
                max_iter = max(1, remaining_budget // 20)
                
                res = minimize(safe_func, x_start, method='L-BFGS-B', bounds=[(-5, 5)] * n,
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success or res.fun < best_f:
                    # The safe_func already updated best_x and best_f
                    pass
            except Exception:
                # If LBFGS-B fails or times out, we keep the best found so far
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
