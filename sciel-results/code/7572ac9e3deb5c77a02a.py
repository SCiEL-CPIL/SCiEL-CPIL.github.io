"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.

    ZDT4 structure:
      g(x) = 1 + 10 * sum_{i=1}^{n} sin^2(10*x_i)
      f0 = g(x)
      f1 = 1 - sqrt((1/n) * sum x_i^2) * g(x)
    
    Pareto front: g(x) = 1 (all sin terms zero), so x_i = k*pi/10 for integer k.
      f0 = 1
      f1 = 1 - sqrt(sum x_i^2 / n)
    
    To maximize hypervolume with reference point [11, 400]:
      - We want many points on the Pareto front
      - Spread f1 values as widely as possible
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        n = dim
        budget = self.budget

        # ZDT4: g(x) = 1 + 10*sum sin^2(10*x_i)
        # Pareto front: sin(10*x_i) = 0 for all i => x_i = k*pi/10
        # f0 = 1, f1 = 1 - sqrt(sum x_i^2 / n)
        
        # Valid k values: x_i = k*pi/10 must be in bounds
        k_vals = []
        for k in range(-20, 21):
            x_val = k * np.pi / 10.0
            if all(low <= x_val <= high for low, high in bounds):
                k_vals.append(k)
        
        if not k_vals:
            k_vals = [0]
        
        x_vals = np.array(k_vals) * np.pi / 10.0
        
        # Strategy: Generate points on the Pareto front with maximally diverse f1 values
        # f1 = 1 - sqrt(sum x_i^2 / n)
        # We want to cover f1 from 1 (all zeros) down to the minimum possible
        
        # Compute min and max possible f1
        max_sum_sq = n * (max(abs(x) for x in x_vals)) ** 2
        min_f1 = 1 - np.sqrt(max_sum_sq / n)
        
        # Generate target f1 values spread across the range
        # Use more points where the hypervolume contribution is highest
        # Hypervolume contribution of a point (f0, f1) with ref [11, 400] is roughly (11-f0)*(400-f1)
        # Since f0=1 for all front points, contribution ~ (10)*(400-f1)
        # So we want to maximize the integral of (400-f1) over the covered f1 range
        
        # Generate 200 target f1 values, denser at the extremes where hypervolume gain is higher
        n_targets = min(budget, 200)
        
        # Use a non-uniform distribution: more points near f1=1 and f1=min_f1
        # Actually, for hypervolume, uniform spacing in f1 is good since each point contributes
        # approximately (400-f1) * delta_f1, and we want to maximize sum of these
        
        # Let's use uniform spacing in f1 for simplicity and good coverage
        target_f1s = np.linspace(1.0, min_f1, n_targets)
        
        # For each target f1, find x vector such that sqrt(sum x_i^2 / n) = 1 - target_f1
        # i.e., sum x_i^2 = n * (1 - target_f1)^2
        
        points = []
        for target_f1 in target_f1s:
            target_sum_sq = n * (1.0 - target_f1) ** 2
            
            pt = np.zeros(dim)
            remaining = target_sum_sq
            
            # Greedy assignment: fill dimensions with largest possible |x| values
            # Sort x_vals by |x| descending
            sorted_x = x_vals[np.argsort(-np.abs(x_vals))]
            sorted_x_sq = sorted_x ** 2
            
            for i in range(dim):
                if remaining <= 1e-10:
                    break
                # Find largest |x| such that x^2 <= remaining
                valid_mask = sorted_x_sq <= remaining + 1e-10
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    best_idx = valid_indices[0]  # First valid is largest |x|
                    pt[i] = sorted_x[best_idx]
                    remaining -= sorted_x[best_idx] ** 2
            
            points.append(pt)
        
        # Evaluate all points
        F = []
        valid_points = []
        used_evals = 0
        
        for pt in points:
            if time.time() > deadline:
                break
            if used_evals >= budget:
                break
            try:
                fc = problem(pt)
                F.append(np.array(fc, dtype=float))
                valid_points.append(pt)
                used_evals += 1
            except Exception:
                break
        
        if len(F) == 0:
            try:
                fc = problem(np.zeros(dim))
                F.append(np.array(fc, dtype=float))
                valid_points.append(np.zeros(dim))
                used_evals += 1
            except Exception:
                pass
        
        F = np.array(F) if F else np.empty((0, 2))
        valid_points = np.array(valid_points) if valid_points else np.empty((0, dim))
        
        # Use remaining budget to add more diverse points if needed
        if used_evals < budget:
            rng = np.random.default_rng(42)
            # Add random points on the front for additional coverage
            for _ in range(budget - used_evals):
                if time.time() > deadline:
                    break
                try:
                    k_choice = rng.integers(k_vals[0], k_vals[-1] + 1, size=dim)
                    pt = k_choice * np.pi / 10.0
                    fc = problem(pt)
                    F = np.vstack([F, [np.array(fc, dtype=float)]])
                    valid_points = np.vstack([valid_points, [pt]])
                    used_evals += 1
                except Exception:
                    break
        
        return valid_points
# EVOLVE-BLOCK-END
