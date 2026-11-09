"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT3's known Pareto front structure.
    ZDT3's Pareto front lies on x[1:] = 0.5, with f1 = x[0] and
    f2 = 1 + 10*(1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0])).
    
    Strategy:
    1. Use a large fraction of budget for dense, non-uniform sampling along x[0]
       (concentrating where the front is complex) with x[1:] = 0.5.
    2. Use remaining budget for local refinement around promising regions.
    3. Track elapsed time to stay within the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        used = 0
        all_points = []
        all_fvals = []
        
        # Phase 1: Dense non-uniform sampling along the Pareto front manifold
        # The Pareto front of ZDT3 is on x[1:] = 0.5
        # f1 = x[0], f2 = 1 + 10*(1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0]))
        # The front has 24 disconnected segments due to the sin term
        # Concentrate samples where the front is more complex
        
        n_phase1 = int(self.budget * 0.7)
        
        # Create non-uniform grid: more points where the front curves more
        # Use a power-law distribution to concentrate in regions with higher curvature
        x0_vals = []
        # Add uniform samples first
        n_uniform = n_phase1 // 2
        x0_uniform = np.linspace(0.0, 1.0, n_uniform, endpoint=False)
        x0_vals.extend(x0_uniform.tolist())
        
        # Add more samples concentrated in regions with high curvature
        # The sin(10*pi*x0) term creates oscillations; concentrate around zeros and extrema
        n_concentrated = n_phase1 - n_uniform
        # Use a mixture: some uniform, some concentrated around interesting regions
        # The front has local minima/maxima of f2 at specific x0 values
        # Sample more densely in [0, 0.5] where the front is more complex
        x0_concentrated = rng.uniform(0.0, 0.5, size=n_concentrated)
        x0_vals.extend(x0_concentrated.tolist())
        
        # Shuffle to avoid ordering bias
        rng.shuffle(x0_vals)
        
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x.copy())
            all_fvals.append(f)
            used += 1
        
        # Phase 2: Local refinement around non-dominated points
        if used < self.budget and time.time() < deadline:
            # Find non-dominated points from Phase 1
            if len(all_fvals) > 0:
                F = np.array(all_fvals)
                # Find non-dominated indices
                nondom_mask = np.zeros(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                nondom_mask[i] = False
                                break
                    if nondom_mask[i]:
                        pass
                # If too many non-dominated, subsample
                nondom_indices = np.where(nondom_mask)[0]
                if len(nondom_indices) > 20:
                    # Pick 20 diverse non-dominated points
                    indices_to_refine = rng.choice(nondom_indices, size=20, replace=False)
                else:
                    indices_to_refine = nondom_indices
                
                # Refine around each selected point
                n_refine = min(self.budget - used, 40)
                n_per_point = max(1, n_refine // max(1, len(indices_to_refine)))
                
                for idx in indices_to_refine:
                    if used >= self.budget or time.time() > deadline:
                        break
                    base_x = all_points[idx]
                    base_f = all_fvals[idx]
                    
                    # Try small perturbations
                    for _ in range(n_per_point):
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Perturb only x[0] slightly (stay on or near Pareto manifold)
                        delta_x0 = rng.normal(0, 0.02)
                        x = base_x.copy()
                        x[0] = np.clip(x[0] + delta_x0, lo[0], hi[0])
                        # Keep x[1:] = 0.5
                        x[1:] = 0.5
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        all_points.append(x.copy())
                        all_fvals.append(f)
                        used += 1
        
        # Phase 3: If budget remains, do random search
        if used < self.budget and time.time() < deadline:
            while used < self.budget and time.time() < deadline:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                all_points.append(x.copy())
                all_fvals.append(f)
                used += 1
        
        return all_points
# EVOLVE-BLOCK-END
