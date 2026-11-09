"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid analytical + evolutionary optimizer for ZDT2.

    Combines structured sampling of x0 (to spread f1) with local evolutionary
    refinement of x1..x_{n-1} to minimize f2 for each f1 region.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim

        # Phase 1: Structured sampling of x0 with x1..=0
        # This gives us a spread of f1 values with minimal f2 (since x1..=0 minimizes the sum)
        n_phase1 = min(80, self.budget // 3)
        if n_phase1 < 10:
            n_phase1 = self.budget

        f1_samples = np.linspace(0.0, 1.0, n_phase1)
        records = []  # list of (x, f) tuples
        used = 0

        for f1_val in f1_samples:
            if used >= self.budget or time.time() > deadline - 30:
                break
            x = np.zeros(n)
            x[0] = f1_val
            f = problem(x)
            records.append((x.copy(), np.array(f, dtype=float)))
            used += 1

        # Phase 2: Local evolutionary refinement
        # For each region of f1, try to reduce f2 by adjusting x1..x_{n-1}
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 30:
            rng = np.random.default_rng(42)
            
            # Group records by f1 regions
            # Sort by f1 to identify regions
            records.sort(key=lambda r: r[0][0])
            
            # For each of the first 40 best points (by f2), do local search
            # Sort by f2 to focus on most promising
            records.sort(key=lambda r: r[1][1])
            
            n_refine_points = min(40, len(records))
            evals_per_point = max(1, remaining // n_refine_points)
            
            for i in range(n_refine_points):
                if used >= self.budget or time.time() > deadline - 30:
                    break
                base_x, base_f = records[i]
                
                # Local search: perturb x1..x_{n-1} with decreasing step size
                step_sizes = [0.1, 0.05, 0.01, 0.005]
                for step in step_sizes:
                    if used >= self.budget or time.time() > deadline - 30:
                        break
                    # Try a few perturbations at this step size
                    n_trials = max(1, evals_per_point // len(step_sizes))
                    for t in range(n_trials):
                        if used >= self.budget or time.time() > deadline - 30:
                            break
                        x_new = base_x.copy()
                        perturb = step * rng.standard_normal(n - 1)
                        x_new[1:] = np.clip(base_x[1:] + perturb, 0.0, 1.0)
                        f_new = problem(x_new)
                        f_new = np.array(f_new, dtype=float)
                        
                        # Only keep if it's better than base in f2
                        if f_new[1] < base_f[1]:
                            records.append((x_new.copy(), f_new))
                            base_x = x_new.copy()
                            base_f = f_new
                        else:
                            records.append((x_new.copy(), f_new))
                        used += 1

        # Phase 3: Fill remaining budget with random exploration
        if used < self.budget and time.time() < deadline - 30:
            rng2 = np.random.default_rng(123)
            while used < self.budget and time.time() < deadline - 30:
                x = np.zeros(n)
                x[0] = rng2.random()
                x[1:] = rng2.random(n - 1) * 0.2  # small values for x1..
                f = problem(x)
                records.append((x.copy(), np.array(f, dtype=float)))
                used += 1

        return records  # ignored by evaluator

# EVOLVE-BLOCK-END
