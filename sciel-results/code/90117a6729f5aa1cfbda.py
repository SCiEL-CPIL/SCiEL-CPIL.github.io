"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import cobyla


class Optimizer:
    """Cobyla-based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 330s wall-clock limit, keep 10s margin
        start_time = time.time()
        deadline = start_time + 320
        
        # State tracking
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Wrapper for Cobyla
        # Cobyla expects a function f(x) and a gradient g(x)
        # Since we don't have gradients, we pass a dummy gradient or use the finite difference approximation?
        # Actually, Cobyla in SciPy handles the gradient internally via finite differences if not provided? 
        # No, SciPy's cobyla requires a gradient function. 
        # However, for black-box optimization where gradients are not available, 
        # we can use a trick: provide a zero gradient or a dummy gradient?
        # Wait, the standard `scipy.optimize.cobyla` signature is:
        # cobyla(func, x0, df, rho0=..., rhobnd=..., maxiter=...)
        # df is the gradient. If we don't have it, we can't use standard cobyla directly without modification.
        
        # Alternative: Use `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell'?
        # The directive suggests a DIFFERENT strategy from (1+1)-ES.
        # The inspiration mentioned "Surrogate-assisted search with RBF interpolation and local Nelder-Mead refinement".
        # Let's try a hybrid: Global search with a few random points to find a good basin, then local refinement with Nelder-Mead.
        # Or, since f11 (Discus) is non-separable and has a "discus" shape, it's smooth but rotated.
        
        # Let's stick to a robust derivative-free method that is different from (1+1)-ES.
        # Nelder-Mead is a simplex method. It is memory-less in terms of population but maintains a simplex.
        # (1+1)-ES maintains a single point and a step size.
        # Nelder-Mead might be better for the "discus" shape if we can get close.
        
        # Strategy:
        # 1. Use a small number of random evaluations to find a promising region.
        # 2. Run Nelder-Mead from that region.
        # 3. If time allows, restart Nelder-Mead from the best point found so far with a smaller initial simplex or different parameters.
        
        # However, the prompt says "Keep the function names and signatures...".
        # And "Python standard library, NumPy and SciPy are available."
        
        # Let's implement a simple restart strategy with Nelder-Mead.
        # Nelder-Mead is often very effective for low-dimensional smooth functions.
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f # Stop evaluating, return best known
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Initial phase: Random search to find a good starting point
        # Budget allocation: 10% for global exploration, 90% for local exploitation
        # But Nelder-Mead uses many function evaluations per iteration.
        # Let's just start Nelder-Mead from a few random points.
        
        # We need to be careful with the budget.
        # Nelder-Mead with dim=5 has a simplex of 6 points.
        # Each iteration involves at least 1 evaluation, sometimes more.
        # Max iterations should be limited.
        
        # Let's try 3 restarts of Nelder-Mead.
        # Each restart gets a portion of the budget.
        
        # However, the "Discus" function is known to be difficult for some optimizers if they get stuck on the flat parts.
        # (1+1)-ES is robust.
        
        # Let's try a different approach: **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy).
        # Is CMA-ES available in SciPy? No, it's in the `cma` package, which is not standard.
        # But we can implement a simplified CMA-ES or use `scipy.optimize.minimize` with method='Powell'?
        # Powell is derivative-free and good for non-separable functions.
        
        # Let's use `scipy.optimize.minimize` with method='Powell'.
        # Powell's method is a line search method.
        
        # Plan:
        # 1. Generate a few random starting points.
        # 2. For each, run Powell's method with a limited number of function evaluations.
        # 3. Keep track of the best result.
        
        # Actually, the inspiration mentioned "RBF interpolation and local Nelder-Mead".
        # Let's implement a simple version of that:
        # 1. Evaluate a small set of random points (e.g., 10-20).
        # 2. Build a simple surrogate (e.g., linear or quadratic regression) or just pick the best point.
        # 3. Run Nelder-Mead from the best point.
        # 4. If budget remains, repeat with the best point found by Nelder-Mead.
        
        # Given the 500 budget, Nelder-Mead can be quite effective.
        # Let's allocate:
        # - 50 random evaluations to find a good basin.
        # - Use the best point as x0 for Nelder-Mead.
        # - Run Nelder-Mead with maxiter such that it doesn't exceed budget.
        
        # How to control Nelder-Mead's function evaluations?
        # SciPy's minimize doesn't have a direct "max_func_evals" parameter for Nelder-Mead, but we can wrap the function to count and raise an exception or return a large value.
        
        # Let's implement a wrapper that stops after a certain number of calls.
        
        def nm_func(x):
            if self.call_count >= self.budget:
                # Return a large value to signal failure/stop, or raise
                # Raising an exception might be cleaner to break out of the loop
                raise ValueError("Budget exceeded")
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # We will run Nelder-Mead in a try-except block.
        
        # Initial random search
        # Let's do 10 random points.
        for i in range(10):
            if self.call_count >= self.budget:
                break
            x0 = np.random.uniform(-5, 5, self.dim)
            try:
                fx = nm_func(x0)
            except ValueError:
                break

        # Now run Nelder-Mead from the best point
        if self.best_x is not None:
            x0 = self.best_x
        else:
            x0 = np.zeros(self.dim)
            
        # We need to be careful with the initial simplex size.
        # Default Nelder-Mead uses a small simplex.
        # For f11, the minimum is in [-4, 4]^5.
        
        # Let's run Nelder-Mead.
        # We'll catch the exception to stop when budget is hit.
        
        try:
            # maxiter is not a strict function call limit, but a loop limit.
            # We rely on the wrapper to stop.
            res = minimize(nm_func, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000})
        except ValueError:
            pass
        except Exception as e:
            # In case of other errors, we stop
            pass

        # If we still have budget, we might try another restart from the best point with a slightly perturbed start?
        # Or just return.
        
        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
