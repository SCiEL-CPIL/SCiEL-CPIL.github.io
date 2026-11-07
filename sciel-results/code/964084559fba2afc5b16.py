"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: radial sweep then aggressive (1+1)-ES for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Dense radial sweep to identify needle axis
        # Use deterministic + random directions
        n_dirs = 30
        # Start with coordinate axes and their negatives
        dirs = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            dirs.append(d)
            dirs.append(-d)
        # Add some random directions
        while len(dirs) < n_dirs:
            d = rng.standard_normal(dim)
            d /= np.linalg.norm(d)
            dirs.append(d)
        dirs = np.array(dirs)

        # For each direction, do a 1D sweep from r=0 to r=4
        # Use adaptive sampling: start coarse, refine where function decreases
        for d in dirs:
            if calls >= budget - 200:  # Reserve budget for phase 2
                break
            
            # Coarse sweep: r = 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
            radii = np.linspace(0, 4.0, 9)
            prev_f = float('inf')
            prev_r = 0.0
            for r in radii:
                if calls >= budget - 200:
                    break
                x = r * d
                f = evaluate(x)
                if f < prev_f:
                    prev_f = f
                    prev_r = r
                else:
                    # Function increased, try to refine between prev_r and r
                    if r - prev_r > 0.3:
                        mid_r = (prev_r + r) / 2
                        x_mid = mid_r * d
                        f_mid = evaluate(x_mid)
                        if f_mid < prev_f:
                            prev_f = f_mid
                            prev_r = mid_r
                    break

        # Phase 2: Aggressive (1+1)-ES
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # Initialize step size
        sigma = 1.0
        s = 0.8
        c = 1.0 / (dim + 1)
        success = 0.0

        while calls < budget:
            # Generate candidate
            mutation = rng.standard_normal(dim)
            candidate = best_x + sigma * mutation
            candidate = np.clip(candidate, -5, 5)
            f_cand = evaluate(candidate)

            if f_cand < best_f:
                # Success
                success += 1.0
                # Adaptive step size: increase on success
                sigma = sigma * (1 + c * (success / max(1, calls - int(calls * 0.5)) - s))
                sigma = max(0.01, min(sigma, 5.0))
            else:
                # Failure: decrease step size
                sigma = sigma * (1 - c * s)
                sigma = max(0.01, min(sigma, 5.0))

            # Reset success counter periodically
            if calls % 20 == 0:
                success = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
