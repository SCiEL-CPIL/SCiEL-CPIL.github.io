"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Optimizer: Global Random Search + Local L-BFGS-B Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Global Search
        # We allocate the first 70% of the budget to random sampling.
        # This helps in finding the global basin for the shifted/rotated ellipsoid.
        global_budget = int(budget * 0.7)
        
        # Initial best is at origin.
        # We will sample points uniformly in [-5, 5]^dim.
        # To improve efficiency, we can use a stratified approach or just simple random.
        # Given dim=5, uniform random is decent.
        
        # We keep track of the best points found during global search to potentially use as starting points for local polish.
        # But since we only have one best_x variable, we just update it.
        
        # To make the global search more effective, we can use a "coarse-to-fine" approach.
        # Or simply random search. Random search is robust to rotation.
        
        # Let's do random search for the first 350 calls.
        # We can also try a few deterministic points like vertices? No, center is usually better for ellipsoids.
        
        # Actually, for step functions, random search might miss the narrow basin if it's very small.
        # But f7 is "Step Ellipsoidal". The step is usually not that tiny in BBOB.
        
        # Let's try a mix: 
        # 1. Random search for 300 calls.
        # 2. Then use L-BFGS-B from the best point found.
        
        # However, L-BFGS-B might get stuck in a local minimum if the function is non-convex (which step functions are).
        # But the basin is convex. If we land in the basin, L-BFGS-B will converge fast.
        
        # Strategy:
        # 1. Evaluate origin.
        # 2. Loop:
        #    - If we have used < 300 calls:
        #       - Generate random point in [-5, 5]^5
        #       - Evaluate
        #    - Else:
        #       - Use L-BFGS-B from best_x
        #       - If L-BFGS-B uses up the budget or fails, stop.
        
        # To improve the random search, we can use a "halving" strategy or just uniform.
        # Let's stick to uniform random for simplicity and robustness.
        
        # Optimization: We can check time in the loop.
        
        global_calls_target = 300
        
        while calls < global_calls_target and time.time() < deadline - 10.0:
            x = rng.uniform(-5.0, 5.0, size=dim)
            _eval(x)
            
        # Phase 2: Local Polish
        # Use L-BFGS-B from the best point found so far.
        # We need to wrap func to count calls and respect the budget.
        
        if calls < budget and time.time() < deadline - 5.0:
            def lbfgs_func(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    # Return a large value to stop optimization
                    return 1e100
                fx = _eval(x)
                if fx is None:
                    return 1e100
                return fx
            
            try:
                res = minimize(lbfgs_func, best_x, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim, options={'maxiter': 50, 'maxfun': budget - calls})
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
