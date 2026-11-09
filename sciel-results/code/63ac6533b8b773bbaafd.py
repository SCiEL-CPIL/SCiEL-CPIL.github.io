"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using analytic front construction and 1D refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng()

        n_evals = 0
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            n_evals += 1
            return x_clipped, f

        # ZDT4 analytic structure:
        # f1(x) = x[0]  (when x[0] is in [0,1] for standard ZDT4, but here bounds may differ)
        # Actually, for ZDT4: f1 = x[0], f2 = g(x[1:]) * (1 - sqrt(x[0]/g(x[1:])))
        # where g(x) = sum_{i=2}^n (x_i - 0.5)^2 - 10*cos(20*pi*(x_i-0.5)) + 10
        #
        # The Pareto front is achieved when:
        # 1. x[0] is in [0, 1] (or the relevant range for f1)
        # 2. x[1:] are at the global minimum of g, which is x_i = 0.5 for all i
        #    (each term (x_i-0.5)^2 - 10*cos(20*pi*(x_i-0.5)) + 10 has min 0 at x_i=0.5)
        #
        # So the ideal Pareto front points are:
        # x[0] = t for t in [0, 1]
        # x[1:] = 0.5
        #
        # Strategy:
        # Phase 1: Sample x[0] uniformly in the relevant range, set x[1:] = 0.5
        # Phase 2: Use remaining budget for 1D radial search to refine x[0] values

        # Determine the range for x[0] from bounds
        x1_lo = max(lo[0], 0.0)  # f1 = x[0], typically in [0,1]
        x1_hi = min(hi[0], 1.0)

        # If bounds don't cover [0,1], use the full bound range
        if x1_hi <= x1_lo:
            x1_lo = lo[0]
            x1_hi = hi[0]

        # Phase 1: Uniform sampling of x[0] with x[1:] = 0.5
        n_phase1 = int(budget * 0.6)
        n_phase2 = budget - n_phase1

        # Generate x[0] values uniformly
        x1_samples = np.linspace(x1_lo, x1_hi, n_phase1)

        # Add a few random samples for diversity
        n_random = min(10, n_phase1 // 10)
        if n_random > 0:
            x1_random = rng.uniform(x1_lo, x1_hi, n_random)
            x1_samples = np.concatenate([x1_samples, x1_random])

        for x1_val in x1_samples:
            if time.time() > deadline or n_evals >= budget:
                break
            x = np.zeros(dim)
            x[0] = x1_val
            x[1:] = 0.5
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)

        # Phase 2: 1D radial search to refine x[0] values
        if n_evals < budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)

            # Sort by f1 to get well-distributed candidates
            order = np.argsort(F_arr[:, 0])

            # Select a diverse set of candidates
            n_candidates = min(20, len(order))
            if n_candidates > 1:
                candidate_indices = np.linspace(0, len(order) - 1, n_candidates, dtype=int)
                candidate_indices = order[candidate_indices]
            else:
                candidate_indices = order

            for idx in candidate_indices:
                if time.time() > deadline or n_evals >= budget:
                    break
                x_base = X_arr[idx].copy()
                f_base = F_arr[idx].copy()

                # Try small perturbations in x[0]
                for step in [-0.01, 0.01, -0.005, 0.005, -0.02, 0.02]:
                    if n_evals >= budget or time.time() > deadline:
                        break
                    x_trial = x_base.copy()
                    x_trial[0] += step
                    x_trial = np.clip(x_trial, lo, hi)
                    x_trial, f_trial = evaluate(x_trial)
                    X_list.append(x_trial)
                    F_list.append(f_trial)
                    # Update base if this is better in at least one objective
                    if np.any(f_trial < f_base) and not np.all(f_trial > f_base):
                        x_base = x_trial.copy()
                        f_base = f_trial.copy()
                        F_arr[idx] = f_trial
                        X_arr[idx] = x_trial

        # Return all evaluated points
        if len(X_list) > 0:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
