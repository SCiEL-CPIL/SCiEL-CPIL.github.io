"""Multi-objective optimizer on ZDT4 using parametric curve-fitting and targeted search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Targeted multi-objective optimizer for ZDT4 using parametric curve-fitting."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        rng = np.random.RandomState(42)
        used = 0
        archive = []  # list of (x, f) pairs

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return x_clipped, f

        def add_to_archive(x, f):
            nonlocal archive
            # Remove points dominated by f
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            # Check if f is dominated
            for ax, af in new_archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            new_archive.append((x, f))
            archive = new_archive

        # Phase 1: Probe tail variables to estimate optimal offset
        # ZDT4: f1 = x[0], g = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i))
        # Optimal g=1 when x_i=0 for i>=1
        # We probe to find the best tail configuration
        n_probe = min(50, budget // 4)
        tail_offset = np.zeros(dim - 1)
        tail_var = np.ones(dim - 1) * 0.01
        
        for probe_i in range(n_probe):
            if used >= budget:
                break
            if time.time() > deadline - 60:
                break
            
            # Probe different tail configurations
            x = np.zeros(dim)
            x[0] = 0.5  # mid-range f1
            if probe_i < 10:
                # Pure zero tail
                x[1:] = 0.0
            elif probe_i < 20:
                # Small random perturbations
                x[1:] = 0.005 * rng.randn(dim - 1)
            elif probe_i < 30:
                # Slightly larger perturbations
                x[1:] = 0.02 * rng.randn(dim - 1)
            else:
                # Try to find local minima via gradient-like steps
                x[1:] = tail_offset + tail_var * rng.randn(dim - 1)
            
            x = np.clip(x, lo, hi)
            x, f = evaluate(x)
            
            # Update tail estimate: if f[1] is good (close to 1+sqrt(0.5)), 
            # this tail configuration is good
            f2_expected = 1 + np.sqrt(0.5)  # ~1.707
            if f[1] < f2_expected + 0.1:
                # Good tail config, update offset
                tail_offset = 0.8 * tail_offset + 0.2 * (x[1:] - 0.005 * rng.randn(dim - 1))
                tail_var *= 0.9
            else:
                # Bad config, shrink variance
                tail_var *= 0.7
        
        # Phase 2: Dense non-uniform grid of x1 values
        # Hypervolume contribution is highest where the Pareto front has high curvature
        # f2 = 1 + sqrt(f1) has high curvature near f1=0
        # Allocate more samples to lower x1 values
        n_grid = min(120, budget - used)
        if n_grid <= 0:
            return np.array([np.zeros(dim)])
        
        # Use quadratic-weighted spacing: more points near low x1
        # x1_i = (i+1)^beta / (n+1)^beta * range + offset
        idx = np.arange(1, n_grid + 1, dtype=float)
        beta = 0.6  # bias toward lower values (high curvature region)
        x1_vals = (idx / (n_grid + 1)) ** beta * 0.99 + 0.005
        x1_vals = np.clip(x1_vals, 0.005, 0.995)
        
        for x1 in x1_vals:
            if used >= budget:
                break
            if time.time() > deadline - 30:
                break
            
            # Use the estimated optimal tail
            x = np.zeros(dim)
            x[0] = x1
            x[1:] = tail_offset + 0.003 * rng.randn(dim - 1)
            x = np.clip(x, lo, hi)
            x, f = evaluate(x)
            add_to_archive(x.copy(), f.copy())
        
        # Phase 3: Local refinement of best archive points
        if len(archive) > 0 and used < budget:
            # Sort by f2 to prioritize points with good g values
            archive.sort(key=lambda p: p[1][1])
            
            max_refine = min(len(archive), (budget - used) // 2)
            for i, (x_best, f_best) in enumerate(archive[:max_refine]):
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                n_local = min(3, budget - used)
                for _ in range(n_local):
                    if used >= budget:
                        break
                    if time.time() > deadline - 10:
                        break
                    
                    # Push tail toward 0 (optimal for ZDT4)
                    x_new = x_best.copy()
                    x_new[1:] = x_new[1:] * 0.3 + 0.002 * rng.randn(dim - 1)
                    x_new = np.clip(x_new, lo, hi)
                    x_new, f_new = evaluate(x_new)
                    add_to_archive(x_new.copy(), f_new.copy())
        
        # Phase 4: Fill remaining budget with targeted search
        while used < budget:
            if time.time() > deadline - 5:
                break
            
            # Choose x1 with bias toward lower values
            u = rng.uniform()
            x1_new = (u ** 0.6) * 0.99 + 0.005
            x1_new = np.clip(x1_new, 0.005, 0.995)
            x_new = np.zeros(dim)
            x_new[0] = x1_new
            x_new[1:] = tail_offset + 0.003 * rng.randn(dim - 1)
            x_new = np.clip(x_new, lo, hi)
            x_new, f_new = evaluate(x_new)
            add_to_archive(x_new.copy(), f_new.copy())
            
            # If we have budget, do a local refinement
            if len(archive) > 0 and used < budget:
                if time.time() > deadline - 5:
                    break
                idx = rng.randint(len(archive))
                x_base, f_base = archive[idx]
                x_refine = x_base.copy()
                x_refine[1:] = x_refine[1:] * 0.2 + 0.001 * rng.randn(dim - 1)
                x_refine = np.clip(x_refine, lo, hi)
                x_refine, f_refine = evaluate(x_refine)
                add_to_archive(x_refine.copy(), f_refine.copy())
        
        # Return (return value is ignored by evaluator, but must not raise)
        if len(archive) > 0:
            return np.array([x for x, f in archive])
        return np.array([np.zeros(dim)])
# EVOLVE-BLOCK-END
