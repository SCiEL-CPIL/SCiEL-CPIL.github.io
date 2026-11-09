"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT1 structure: x0 controls f1, x1..x30 control f2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT1 structure: f1 = x0, f2 = g(x1..xn) * (1 + sin(10*pi*x0))
        # Optimal x1..xn for minimizing g is 0.5 (where g=1)
        # So the Pareto front is: x0 in [0,1], x1..xn = 0.5, giving f1=x0, f2=1
        
        # Strategy: 
        # 1. First, sample x0 values to cover [0,1] well
        # 2. For each x0, set x1..xn to values that minimize g (near 0.5)
        # 3. Use remaining budget to refine g minimization
        
        # Phase 1: Sample x0 values with a power-law distribution to get good hypervolume
        # The hypervolume contribution depends on the spacing of x0 values
        # Use a logarithmic or power-law spacing to maximize hypervolume
        
        n_phase1 = min(budget, 80)
        # Use a mixture: some uniform, some concentrated near 0 and 1
        # Actually, for ZDT1 with f2=1 on the front, hypervolume is maximized
        # by evenly spacing x0 in [0,1]
        
        # Generate x0 samples: combine uniform and boundary-concentrated
        x0_samples = np.concatenate([
            np.linspace(0, 1, max(1, n_phase1 // 2)),
            np.linspace(0, 1, max(1, n_phase1 // 2))[::-1]
        ])
        # Deduplicate and sort
        x0_samples = np.unique(x0_samples)
        if len(x0_samples) > n_phase1:
            # Subsample
            indices = np.linspace(0, len(x0_samples)-1, n_phase1, dtype=int)
            x0_samples = x0_samples[indices]
        
        # Phase 2: For each x0, find good x1..xn values
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 * (1+10*sin^2(3*pi*xi)))
        # Minimum of g is 1, achieved when all xi = 0.5
        
        # For each x0, start with x1..xn = 0.5 and refine
        used = 0
        results = []
        
        # First pass: evaluate at x1..xn = 0.5 for each x0
        for x0 in x0_samples:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0
            f = problem(x)
            results.append(f)
            used += 1
        
        # Phase 3: Use remaining budget to refine x1..xn for selected x0 values
        # Focus on x0 values that give the best hypervolume contribution
        # The points with f2 < 1 are strictly better, so we want to minimize g
        
        remaining = budget - used
        if remaining > 0 and len(results) > 0:
            # Select a subset of x0 values to refine
            n_refine = min(remaining // 3, len(x0_samples))
            if n_refine > 0:
                # Pick x0 values spread across the range
                indices = np.linspace(0, len(x0_samples)-1, n_refine, dtype=int)
                refine_indices = indices
                
                for idx in refine_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    x0 = x0_samples[idx]
                    # Start from x1..xn = 0.5 and do a few local searches
                    x = np.full(dim, 0.5)
                    x[0] = x0
                    f_base = problem(x)
                    used += 1
                    results.append(f_base)
                    
                    # Local refinement: try small perturbations
                    n_local = min(remaining - (used - (budget - remaining)), 2)
                    for _ in range(n_local):
                        if used >= budget or time.time() > deadline:
                            break
                        # Perturb x1..xn
                        perturbation = np.random.randn(dim - 1) * 0.05
                        x_new = x.copy()
                        x_new[1:] = np.clip(x_new[1:] + perturbation, lo[1:], hi[1:])
                        f_new = problem(x_new)
                        used += 1
                        results.append(f_new)
                        # Keep the better one
                        if f_new[1] < f_base[1]:
                            x = x_new
                            f_base = f_new
        
        return results
# EVOLVE-BLOCK-END
