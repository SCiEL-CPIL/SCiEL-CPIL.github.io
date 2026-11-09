"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """L-BFGS-B based optimizer with multi-start global exploration for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit, use 280s to be safe for Python overhead
        deadline = start_time + 280.0

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)
        
        # Store evaluated points for restarts
        # List of (f_val, x_vec)
        history = []

        def safe_func(x):
            nonlocal calls, best_f, best_x
            # Check time
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            # Check budget
            if calls >= total_budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            
            # Check for finite values
            if not np.all(np.isfinite(x_clipped)):
                return 1e10

            f = func(x_clipped)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
                
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # We allocate 20% of the budget to exploration to find the basin.
        # For n=5, random sampling is effective.
        n_explore = max(10, int(total_budget * 0.2))
        
        # Generate diverse starting points
        # 1. Origin
        starts = [np.zeros(n)]
        # 2. Deterministic points near boundaries (shifts can be anywhere in [-4,4])
        starts.append(np.ones(n) * 3.0)
        starts.append(np.ones(n) * -3.0)
        starts.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0]))
        
        # 3. Random points
        while len(starts) < n_explore:
            starts.append(rng.uniform(-5, 5, size=n))
            
        # Evaluate all exploration points
        for x0 in starts:
            if calls >= total_budget or time.time() > deadline:
                break
            try:
                f0 = safe_func(x0)
                history.append((f0, x0.copy()))
            except (StopIteration, TimeoutError):
                break
                
        if not history:
            return best_x, best_f
            
        # Sort history by fitness
        history.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Use the remaining budget for local optimization.
        # We try the best candidates first.
        # L-BFGS-B is a quasi-Newton method that handles the Rosenbrock valley well.
        
        # We attempt refinement on the top 5 candidates.
        # If one converges and uses up the budget, we stop.
        # If one fails to converge (stagnates), we move to the next candidate.
        
        top_k = min(5, len(history))
        
        for i in range(top_k):
            if calls >= total_budget or time.time() > deadline:
                break
                
            x_start = history[i][1]
            
            # Skip if we have already converged very well (optional optimization)
            # But L-BFGS-B is cheap in iterations if close, so just run it.
            
            try:
                # maxiter is set high, but safe_func will raise StopIteration 
                # if the call budget is exceeded.
                # ftol and gtol are set tight to ensure convergence.
                res = minimize(
                    safe_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 200, 
                        'ftol': 1e-15,
                        'gtol': 1e-10,
                        'disp': False,
                        'maxls': 20 # Limit line search steps to avoid wasting budget
                    }
                )
                # safe_func updates best_x and best_f internally
            except (StopIteration, TimeoutError):
                # Budget or time exceeded, stop all further optimizations
                break
            except Exception:
                # If L-BFGS-B fails for other reasons, continue to next candidate
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
