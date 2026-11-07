"""Multi-start local optimizer with Sobol exploration and L-BFGS-B refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol + L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol Space-Filling Exploration ---
        # Use 64 points for initial coverage.
        n_sobol = 64
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0
            
            # Store evaluations to find top K
            evals = []
            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                f_val = wrapped_func(x)
                evals.append((f_val, x))
        except Exception:
            evals = []
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                f_val = wrapped_func(x)
                evals.append((f_val, x))

        # Sort by function value to get top candidates
        if evals:
            evals.sort(key=lambda item: item[0])
            # Keep top 5 distinct points for multi-start
            top_candidates = [x for f, x in evals[:5]]
        else:
            top_candidates = [best_x]

        bounds = [(-5.0, 5.0)] * dim
        
        # --- Phase 2: Multi-start Local Refinement ---
        # We will iterate through candidates and run L-BFGS-B.
        # If we get stuck, we generate new candidates.
        
        # List of starting points for local search
        start_queue = list(top_candidates)
        
        # Track the best result from local searches
        local_best_f = best_f
        local_best_x = best_x
        
        # Stagnation counter
        stagnation_count = 0
        max_stagnation = 3 # If 3 local searches in a row don't improve global best, we need new starts
        
        # We will loop until budget or time is up
        # We add new random starts to the queue as we go
        
        while eval_count < budget and time.time() < deadline:
            # If queue is empty, generate new random starts
            if not start_queue:
                # Generate 5 new random points
                for _ in range(5):
                    x_new = rng.uniform(-5.0, 5.0, dim)
                    start_queue.append(x_new)
                # Also add a perturbation of the current best
                x_pert = best_x + rng.normal(0, 0.5, dim)
                x_pert = np.clip(x_pert, -5.0, 5.0)
                start_queue.append(x_pert)
            
            # Pop a start point
            x_start = start_queue.pop(0)
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                # Check if this local search improved our global best
                if res.fun < best_f - 1e-10:
                    best_f = res.fun
                    best_x = res.x
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # If we are stagnating, we might want to add more diverse points to the queue
                if stagnation_count >= max_stagnation:
                    # Add a few Sobol-like points from a new random seed to escape local minima
                    for _ in range(3):
                        x_esc = rng.uniform(-5.0, 5.0, dim)
                        start_queue.append(x_esc)
                    stagnation_count = 0
                    
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # If we have time and budget, do one more high-precision L-BFGS-B from the absolute best
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
