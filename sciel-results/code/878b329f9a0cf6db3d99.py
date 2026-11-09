"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sobol quasi-random sweep with local Gaussian refinement for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin: kill at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        d = self.dim
        
        # Phase 1: Deterministic Sobol quasi-random sequence for uniform coverage
        # We want to use about 60% of the budget for the initial sweep
        n_sweep = int(n * 0.6)
        
        # Generate Sobol sequence using scipy
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True)
            # Need power of 2 for Sobol; round up
            n_points = 1
            while n_points < n_sweep:
                n_points *= 2
            samples = sampler.random(n_points)[:n_sweep]
        except Exception:
            # Fallback to Halton or random if scipy qmc fails
            samples = np.random.rand(n_sweep, d)
        
        # Map to bounds
        X_sweep = lo + (hi - lo) * samples
        F_sweep = np.array([problem(x) for x in X_sweep])
        
        eval_count = n_sweep
        
        # Phase 2: Local refinement around non-dominated points
        # Identify non-dominated points from the sweep
        def is_non_dominated(F):
            """Return boolean mask of non-dominated points."""
            n_pts = len(F)
            mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not mask[i]:
                    continue
                for j in range(n_pts):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask
        
        # Find non-dominated points
        nd_mask = is_non_dominated(F_sweep)
        nd_indices = np.where(nd_mask)[0]
        
        # If too many or too few, adjust
        if len(nd_indices) == 0:
            # Fallback: use all points
            nd_indices = np.arange(n_sweep)
        
        # Budget for refinement
        remaining_budget = n - eval_count
        
        # Strategy: distribute remaining budget among top non-dominated points
        # We'll pick up to 20 of the best non-dominated points (by hypervolume contribution or just random subset)
        # For simplicity, let's pick a diverse subset. Since we don't have a full HV calculation here easily,
        # we can just take the first few or a random sample of the non-dominated set.
        # To ensure diversity, let's take a random sample of min(20, len(nd_indices)) points.
        n_refine_centers = min(20, len(nd_indices))
        if n_refine_centers > 0:
            # Randomly select centers from the non-dominated set
            center_indices = np.random.choice(nd_indices, size=n_refine_centers, replace=False)
            centers = X_sweep[center_indices]
            
            # Distribute remaining budget among centers
            n_per_center = max(1, remaining_budget // n_refine_centers)
            # Adjust for remainder
            n_per_center_list = [n_per_center] * n_refine_centers
            rem = remaining_budget - sum(n_per_center_list)
            for i in range(rem):
                n_per_center_list[i] += 1
            
            # Gaussian refinement
            # Adaptive sigma: start with 0.1 * range, shrink over time
            base_sigma = 0.1 * (hi - lo)
            
            for c_idx, center in enumerate(centers):
                n_evals = n_per_center_list[c_idx]
                if n_evals <= 0:
                    continue
                
                # Generate Gaussian perturbations
                # We can use a shrinking sigma for better local search
                # Simple approach: fixed small sigma
                sigma = base_sigma * 0.5
                
                # Generate points
                # To save time, generate in batches
                X_local = center + np.random.randn(n_evals, d) * sigma
                X_local = np.clip(X_local, lo, hi)
                
                # Evaluate
                for x in X_local:
                    if eval_count >= n:
                        break
                    if time.time() > deadline - 5:
                        break
                    problem(x)
                    eval_count += 1
        
        # Return the first point (ignored by evaluator, but must be valid)
        if len(X_sweep) > 0:
            return X_sweep[0]
        return lo

# EVOLVE-BLOCK-END
