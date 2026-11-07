"""Multi-objective optimizer for WFG4 with Pareto-front-biased sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """WFG4 optimizer: Pareto-front-biased sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 260.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        range_ = hi - lo

        rng = np.random.default_rng(42)

        # Strategy: Pareto-front-biased sampling
        # For WFG4 3-objective: front is parameterized by t in [0,1]
        # f1 = t, f2 = (1-t)*cos(0.5*pi*t), f3 = (1-t)*sin(0.5*pi*t)
        # x1 should be t (mapped to bounds), remaining vars set for convergence

        # Map t in [0,1] to x1 in [lo[0], hi[0]]
        # The "convergence" variables (x2..x12) should be set to values
        # that minimize the distance to the Pareto front.
        # Typically for WFG, setting them to 0 or 0.5 (midpoint) works well.
        # Let's use the midpoint of bounds for convergence vars as a good default.
        mid = (lo + hi) / 2.0

        used = 0
        all_F = []  # list of objective vectors
        all_X = []  # list of decision vectors

        def evaluate_and_store(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=np.float64)
            all_F.append(f)
            all_X.append(x.copy())
            used += 1
            return f

        # Phase 1: Dense Pareto-front-biased sampling
        # Use ~70% of budget for front sampling
        n_front = int(budget * 0.70)
        n_front = max(n_front, 10)

        # Sample t values: mix of uniform and slightly concentrated
        # Use a combination: uniform + some clustered
        t_vals = np.linspace(0.0, 1.0, n_front, endpoint=True)
        # Add some random perturbation for robustness
        t_vals = np.clip(t_vals + rng.normal(0, 0.005, n_front), 0.0, 1.0)

        for t in t_vals:
            if used >= budget or time.time() > deadline:
                break

            x = mid.copy()
            # Set x1 to map t to the first variable's range
            x[0] = lo[0] + t * range_[0]

            # For WFG4, the remaining variables should be set to minimize
            # the "shape" component. Setting to midpoint is a reasonable default.
            # But let's also try setting them to 0 (in normalized space) or specific values.
            # Actually, for WFG4, the convergence variables should be at values
            # that make the transition function output 0. This typically means
            # setting them to specific values. Let's use midpoint as a safe default.

            evaluate_and_store(x)

        # Phase 2: Local refinement around front points
        # Use remaining budget to do small perturbations around front points
        n_refine = budget - used
        if n_refine > 0 and all_F:
            F_arr = np.array(all_F)

            # Find non-dominated points
            # Simple dominance check
            n_pts = len(F_arr)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break

            # Pick non-dominated points for refinement
            nd_idx = np.where(~is_dominated)[0]
            if len(nd_idx) > 0:
                # Shuffle and pick some
                rng.shuffle(nd_idx)
                n_to_refine = min(n_refine, len(nd_idx) * 3)  # up to 3 refinements per ND point
                for idx in nd_idx[:n_to_refine // 3 if n_to_refine > 0 else 0]:
                    if used >= budget or time.time() > deadline:
                        break

                    base_x = all_X[idx]
                    # Small perturbations
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        # Perturb only convergence variables (x2..x12)
                        x = base_x.copy()
                        noise = rng.normal(0, 0.05, dim)
                        x[1:] = np.clip(x[1:] + noise * range_[1:], lo[1:], hi[1:])
                        evaluate_and_store(x)

        # Phase 3: If budget remains, do some random exploration
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = lo + rng.random(dim) * range_
                evaluate_and_store(x)

        return all_X
# EVOLVE-BLOCK-END
