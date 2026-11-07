"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Nelder-Mead optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Nelder-Mead parameters
        # Initial simplex size
        init_step = 1.0
        
        # We will perform multiple restarts of Nelder-Mead
        # Each restart uses a portion of the budget.
        # Nelder-Mead is a local method, so restarts help escape local minima.
        # For dim=5, a reasonable maxiter for one NM run is around 50-100 evaluations.
        # Let's use a budget-aware maxiter.
        
        # Strategy: 
        # 1. Start with a random point.
        # 2. Run Nelder-Mead with a limited budget (e.g., 50% of remaining).
        # 3. If it converges or budget runs out, restart near the best found.
        # 4. Repeat until total budget is exhausted or deadline reached.

        # To strictly respect the call budget and deadline, we wrap the func
        # and check the time/call count inside.
        
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x, deadline
            if calls_made >= self.budget or time.time() > deadline:
                # Return a large value to stop optimization, but we need to be careful
                # scipy might not stop immediately if we just return inf.
                # We'll raise an exception to break out of scipy's loop?
                # No, raising exceptions can be messy. 
                # Better: return a very large number and rely on maxiter.
                # But if it's still running, it wastes time.
                # Let's stick to a strict maxiter based on remaining budget.
                pass
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # We need to be careful with scipy.optimize.minimize. 
        # It doesn't natively support a "stop after N calls" easily without a callback that raises.
        # We can use a callback to check time/budget and raise a specific exception.
        
        class BudgetExhausted(Exception):
            pass

        def check_budget(x, fx):
            nonlocal calls_made, deadline
            if calls_made >= self.budget or time.time() > deadline:
                raise BudgetExhausted()

        # Initial point
        x0 = rng.uniform(-5, 5, self.dim)
        fx0 = wrapped_func(x0)
        
        # Simplex options for Nelder-Mead
        # initial_simplex: array of shape (dim+1, dim)
        # We can let scipy generate it, or specify it.
        # Specifying it might be faster/more predictable.
        
        while calls_made < self.budget and time.time() < deadline:
            # Determine how many evaluations we can afford for this NM run
            # NM typically uses ~ dim * maxiter evaluations.
            # Let's reserve some budget for potential future restarts.
            # If we have 500 budget, and dim=5.
            # A good NM run might take 100-200 evals.
            # Let's try to use up to 80% of remaining budget for one run, 
            # but cap maxiter to ensure we don't overshoot.
            
            remaining_budget = self.budget - calls_made
            if remaining_budget <= 0:
                break
            
            # Estimate maxiter. 
            # NM iterations consume variable number of points.
            # Heuristic: maxiter = remaining_budget / (dim + 1)
            # This is a safe upper bound.
            maxiter = max(10, int(remaining_budget / (self.dim + 1)))
            
            # If we have very little budget left, do a local search manually?
            # Or just one more NM step?
            
            # Create initial simplex around current best
            # If we have a best_x, center there. Else use x0.
            center = best_x if best_x is not None else x0
            
            # Generate initial simplex
            # Standard simplex: center + step * basis vectors
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = center
            for i in range(self.dim):
                simplex[i+1] = center
                simplex[i+1][i] += init_step
                simplex[i+1] = np.clip(simplex[i+1], -5, 5)
            
            # Ensure simplex points are distinct and valid
            # Check if any points are identical due to clipping
            for i in range(1, self.dim + 1):
                if np.allclose(simplex[0], simplex[i]):
                    simplex[i][i] += init_step * 0.1
                    simplex[i] = np.clip(simplex[i], -5, 5)

            try:
                # Run Nelder-Mead
                # method='Nelder-Mead'
                # options: 
                #   'initial_simplex': simplex
                #   'maxiter': maxiter
                #   'xatol': 1e-6
                #   'fatol': 1e-6
                #   'adaptive': True (helps with scaling)
                
                # Note: scipy's Nelder-Mead calls func for each new point.
                # Our wrapped_func tracks calls.
                # The callback checks for budget exhaustion.
                
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    center, 
                    method='Nelder-Mead', 
                    options={
                        'initial_simplex': simplex,
                        'maxiter': maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'callback': check_budget
                    }
                )
                
                # If res.success is False due to maxiter, that's fine.
                # If it converged, even better.
                
            except BudgetExhausted:
                # Budget ran out during the optimization
                break
            except Exception as e:
                # If any other error occurs, break to be safe
                break

            # If we finished a NM run without exhausting budget, 
            # we should restart.
            # To avoid infinite loops or getting stuck, we jitter the center.
            if best_x is not None:
                # Jitter around best
                jitter = rng.uniform(-0.5, 0.5, self.dim)
                center = best_x + jitter
                center = np.clip(center, -5, 5)
            else:
                center = rng.uniform(-5, 5, self.dim)
            
            # Small pause to let time update? No, time.time() is fast enough.
            
            # If we have very few calls left, break
            if self.budget - calls_made < (self.dim + 1):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
