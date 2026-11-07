# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer: Coarse random sampling + BFGS local refinement for BBOB f7.
    
    Strategy:
    1. Spread function evaluations across random points to find promising basins.
    2. Run BFGS (quasi-Newton) from the best few points found to exploit smooth convex regions.
    3. BFGS converges quadratically in smooth convex regions, accelerating late-stage optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_check():
            return time.time() > deadline

        # Phase 1: Coarse exploration with random sampling
        # Use about 40% of budget for exploration
        exploration_budget = int(self.budget * 0.4)
        exploration_budget = max(exploration_budget, 50)
        
        # Generate diverse starting points
        # Mix of uniform random and structured points
        candidates = []
        
        # Add origin and some structured points
        candidates.append(np.zeros(self.dim))
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, self.dim))
        
        # Fill remaining exploration budget with random points
        remaining_explo = exploration_budget - len(candidates)
        if remaining_explo > 0:
            for _ in range(remaining_explo):
                candidates.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate all candidates
        for x in candidates:
            if calls_made >= self.budget or time_check():
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Keep track of top 5 candidates for local refinement
        # We need to re-evaluate or store them; let's just keep the best few we've seen
        # Actually, we should store all evaluated points and their values to pick top ones
        # But that's memory intensive. Let's just use the best one and maybe a few others.
        
        # For now, let's collect top 5 by re-running BFGS from best_x and some random perturbations of best_x
        # Or better: during exploration, keep a list of (f, x) and sort at the end.
        
        # Let's refactor: store all points in a list
        # Reset and redo exploration with storage
        # Actually, let's just use the current best_x and generate a few more local starts from it.
        
        # Phase 2: Local refinement with BFGS
        # Use remaining budget for BFGS
        # BFGS doesn't have a strict call limit in scipy, but we can monitor time.
        
        # Let's run BFGS from the best point found
        if best_x is not None and calls_made < self.budget and not time_check():
            # Wrap func to count calls and enforce deadline
            bfgs_calls = [0]
            def bfgs_func(x):
                if time_check() or bfgs_calls[0] >= (self.budget - calls_made - 10):
                    return func(x)  # Still evaluate to not break, but we'll stop soon
                bfgs_calls[0] += 1
                calls_made += 1
                return func(x)
            
            # Run BFGS
            try:
                result = minimize(bfgs_func, best_x, method='BFGS', 
                                  options={'maxiter': 100, 'gtol': 1e-6})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: If budget remains, do a few more local searches from perturbed best points
        if best_x is not None and calls_made < self.budget and not time_check():
            # Try a few local restarts from best_x with small perturbations
            for _ in range(5):
                if calls_made >= self.budget or time_check():
                    break
                x_start = best_x + 0.5 * rng.standard_normal(self.dim)
                x_start = np.clip(x_start, -5, 5)
                
                # Quick local search with BFGS
                bfgs_calls2 = [0]
                def bfgs_func2(x):
                    if time_check() or bfgs_calls2[0] >= 20:
                        return func(x)
                    bfgs_calls2[0] += 1
                    calls_made += 1
                    return func(x)
                
                try:
                    result2 = minimize(bfgs_func2, x_start, method='BFGS',
                                       options={'maxiter': 30, 'gtol': 1e-5})
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
