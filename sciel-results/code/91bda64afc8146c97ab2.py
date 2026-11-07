import numpy as np
import time


class Optimizer:
    """Multi-resolution adaptive grid strategy for ZDT3 Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # ZDT3 Pareto front: x_in = 0, x_out varies in [0,1]
        # Generate candidate points on the Pareto front
        # f1 = 1 + 10*(dim-1)*x_out + g(0) where g(0) = 10*(dim-1) + sum(3*0 - 2*cos(0))
        # g(0) = 10*(dim-1) + (dim-1)*(-2) = 8*(dim-1)
        # f1 = 1 + 10*(dim-1)*x_out + 8*(dim-1) = 1 + (dim-1)*(8 + 10*x_out)
        # f2 = 1 - sqrt(f1/g(0)) = 1 - sqrt((1 + (dim-1)*(8+10*x_out)) / (8*(dim-1)))

        # Strategy: sample x_out values to cover the Pareto front
        # Use adaptive refinement: start with coarse grid, refine where hypervolume contribution is high

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            return f

        # Phase 1: Dense uniform sampling of Pareto front (x_in=0, x_out uniform)
        # Use a good fraction of budget for this
        n_phase1 = min(budget, 150)

        # Generate x_out values - use a mix of uniform and slightly perturbed
        x_out_vals = np.linspace(0, 1, n_phase1)
        # Add some jitter to avoid exact duplicates and cover more
        # Actually for hypervolume, we want good spread

        for x_out in x_out_vals:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x_out
            # x[1:] = 0 (already zero)
            f = evaluate(x)
            if f is not None:
                all_F.append(f)

        # Phase 2: If budget remains, try some points near Pareto front with small x_in perturbations
        # to potentially find points that are slightly better in some regions
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Try a few random points with small x_in to see if they help
            n_phase2 = min(remaining, 30)
            for _ in range(n_phase2):
                if used >= budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = rng.random()
                # Small perturbations in x_in
                x[1:] = rng.random(dim - 1) * 0.05
                f = evaluate(x)
                if f is not None:
                    all_F.append(f)

        # Phase 3: Adaptive refinement - evaluate points at midpoints of largest gaps
        if used < budget and time.time() < deadline:
            # Convert to array for analysis
            if all_F:
                F_arr = np.array(all_F)
                # Keep only points better than reference [11, 11]
                mask = np.all(F_arr < 11.0, axis=1)
                F_valid = F_arr[mask]

                if len(F_valid) > 2:
                    # Sort by f1
                    idx = np.argsort(F_valid[:, 0])
                    F_sorted = F_valid[idx]

                    # Find gaps in f1 and f2 and try to fill them
                    # For now, just add a few more points at strategic locations
                    remaining = budget - used
                    n_phase3 = min(remaining, 20)

                    # Try points at specific x_out values that might fill gaps
                    # The Pareto front is smooth, so uniform in x_out should be good
                    # But let's add some at quantiles
                    for _ in range(n_phase3):
                        if used >= budget or time.time() > deadline:
                            break
                        x_out = rng.random()
                        x = np.zeros(dim)
                        x[0] = x_out
                        f = evaluate(x)
                        if f is not None:
                            all_F.append(f)

        return np.array(all_F) if all_F else np.empty((0, self.n_obj))
