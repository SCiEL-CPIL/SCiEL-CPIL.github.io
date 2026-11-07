"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Structure-Exploiting Optimizer: Lattice-Primed Multi-Start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        history = []

        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                history.append((x.copy(), f_val))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Rastrigin has global minima at integers.
        # 2. The problem is rotated and shifted.
        # 3. We will exploit the lattice structure by initializing on integer points.
        # 4. We will use a fast, derivative-free local optimizer (Nelder-Mead) for refinement.
        # 5. We will use a "lattice-primed" approach: start at integers, then jump to neighbors if stuck.
        
        # Phase 1: Lattice Sampling
        # We know the minimizer is in [-4, 4]^5.
        # Let's sample a subset of integer points in this range.
        # Total integer points: 9^5 = 59049. Too many.
        # Let's sample a coarser grid first, or random integer points.
        # Since the shift is hidden, we don't know the exact lattice.
        # However, the function is periodic with period 2 (Rastrigin).
        # So minima are at ..., -2, 0, 2, ... in each dimension (before rotation/shift).
        # After rotation/shift, they are at rotated/shifted positions.
        # But the "integer" structure is preserved in the transformed space.
        # So, if we could invert the rotation, we'd sample integers.
        # We can't. But we can sample points that are "integer-like" in the original space? No, we only have f(x).
        
        # Alternative: Just sample random points, but bias towards "low frequency" structures?
        # Actually, the best strategy for rotated Rastrigin is often just good global search + local polish.
        # The current program's weakness might be that Nelder-Mead gets stuck in local minima if the initial point is not close enough to a basin.
        
        # Let's try a different local optimizer: CMA-ES is better for rotated problems.
        # But CMA-ES is slower.
        # How about using a population-based method like DE or PSO for global search, then NM for local?
        
        # Let's try:
        # 1. 100 calls: Random search to find good basins.
        # 2. 400 calls: Split among top 4 basins, running CMA-ES (scipy) with tight budget.
        
        # Wait, the current program already does multi-start NM.
        # The inspiration 1 (score 1.28) uses "Lattice Initialization".
        # What does that mean? Maybe it initializes the simplex at integer points?
        # Or maybe it uses the fact that the gradient of Rastrigin is zero at integers?
        
        # Let's try a simpler, more robust approach:
        # 1. Use scipy.optimize.minimize with method='CMA' for local refinement.
        #    CMA-ES is much more robust to rotation and scaling than NM.
        # 2. Use a multi-start strategy with centers chosen from a grid of "likely" minima locations.
        #    Since we don't know the rotation, we can't know the exact locations.
        #    But we can sample points that are "spread out" and then refine.
        
        # Let's allocate budget:
        # 50 calls for global exploration (random + snapped to 0.5 grid?).
        # 450 calls for local refinement (CMA-ES).
        # Split 450 among 5 best basins -> 90 calls each.
        # CMA-ES with popsize=10, 90 calls is ~9 iterations. Might be enough for 5D.
        
        global_budget = 50
        num_restarts = 5
        local_budget = (budget - global_budget) // num_restarts
        
        # Phase 1: Global Search
        candidates = []
        # 25 random uniform
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            candidates.append(x)
        # 25 random snapped to 0.5 grid (Rastrigin period is 2, so 0.5 is fine)
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            x_snapped = np.round(x * 2) / 2
            x_snapped += rng.normal(0, 0.1, n)
            candidates.append(x_snapped)
            
        rng.shuffle(candidates)
        
        for x in candidates:
            if calls_made >= global_budget or time.time() > deadline:
                break
            evaluate(x)
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = 1000.0
            history.append((best_x, best_f))

        # Phase 2: Local Refinement with CMA-ES
        # Pick top 5 distinct points from history
        history.sort(key=lambda item: item[1])
        
        selected_starts = []
        for x, f in history:
            if len(selected_starts) >= num_restarts:
                break
            # Check if x is too close to already selected
            too_close = False
            for xs in selected_starts:
                if np.linalg.norm(x - xs) < 1.0:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x)
                
        if not selected_starts:
            selected_starts = [best_x]

        # Run CMA-ES from each selected start
        for start_x in selected_starts:
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Define objective for scipy
            def scipy_func(x):
                return evaluate(x)
            
            try:
                # CMA-ES options
                # popsize: number of samples per generation
                # maxiter: max number of generations
                # We have ~90 calls per restart. popsize=10 -> 9 generations.
                # Let's try popsize=8, maxiter=15 (8*15=120 calls, but we cap by budget)
                res = minimize(
                    scipy_func, 
                    start_x, 
                    method='CMA', 
                    options={
                        'popsize': 8,
                        'maxiter': 20,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'tolfun': 1e-4
                    }
                )
            except Exception:
                pass
            
            # Update best
            if calls_made < budget and time.time() < deadline:
                # Check if res.x is better
                f_res = evaluate(res.x)
                
        return best_x, best_f

# EVOLVE-BLOCK-END
