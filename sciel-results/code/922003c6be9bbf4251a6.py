"""Multi-start local optimizer with Sobol exploration and adaptive refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol multi-start + L-BFGS-B + Nelder-Mead optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Store all evaluations for analysis
        all_points = []
        all_fvals = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            all_points.append(x.copy())
            all_fvals.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol Space-Filling Exploration ---
        # Use 64 points for initial coverage to identify diverse basins
        n_sobol = 64
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0
            
            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        # Sort by function value to get top diverse candidates
        if all_points:
            sorted_idx = np.argsort(all_fvals)
            # Keep top 8 distinct points for multi-start
            top_candidates = []
            for idx in sorted_idx[:8]:
                top_candidates.append(all_points[idx].copy())
        else:
            top_candidates = [best_x.copy()]

        bounds = [(-5.0, 5.0)] * dim
        
        # --- Phase 2: Multi-start L-BFGS-B Refinement ---
        # Run L-BFGS-B from each top candidate
        # Use a budget allocation: ~60% of remaining for local searches
        remaining_budget = budget - eval_count
        local_budget_per_start = max(20, remaining_budget // max(1, len(top_candidates)))
        
        for x_start in top_candidates:
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Run L-BFGS-B from this start
            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                # wrapped_func already updates best
            except Exception:
                pass
        
        # --- Phase 3: Adaptive Nelder-Mead Fine-Tuning ---
        # Use Nelder-Mead for final polish on the best point
        # Also try from a few nearby points
        if eval_count < budget and time.time() < deadline:
            # First, polish the global best
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass
        
        # --- Phase 4: Guided Exploration with Local Refinement ---
        # If budget remains, do targeted exploration near promising regions
        # Use a simple strategy: perturb the best point and nearby good points
        if eval_count < budget and time.time() < deadline:
            # Get top 3 points from all evaluations
            if len(all_points) >= 3:
                sorted_idx = np.argsort(all_fvals)[:3]
                guide_points = [all_points[i].copy() for i in sorted_idx]
            else:
                guide_points = [best_x.copy()]
            
            # For each guide point, do a small local search with random perturbations
            for gp in guide_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Try several perturbations of decreasing scale
                for scale in [1.0, 0.5, 0.2, 0.1]:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    
                    x_try = gp + rng.normal(0, scale, dim)
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = wrapped_func(x_try)
                    
                    # If this perturbation improved or is close, do a quick NM refinement
                    if f_try < best_f + 0.1:
                        try:
                            res = minimize(
                                wrapped_func,
                                x_try,
                                method='Nelder-Mead',
                                options={'maxiter': 50, 'xatol': 1e-9, 'fatol': 1e-11}
                            )
                        except Exception:
                            pass
        
        # --- Phase 5: Final High-Precision Polish ---
        # One last L-BFGS-B with tighter tolerances on the absolute best
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
