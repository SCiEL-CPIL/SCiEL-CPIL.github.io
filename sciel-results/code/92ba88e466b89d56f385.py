"""Optimal-control-inspired adaptive hypervolume maximizer for ZDT1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive multi-objective optimizer that allocates its evaluation budget
    non-uniformly across the Pareto front using a recursive bisection /
    marginal-hypervolume-gain strategy.

    Strategy:
      1.  Perform a coarse, stratified sampling of the Pareto front to establish
          an initial set of non-dominated points and estimate the front shape.
      2.  Iteratively select the next evaluation point by maximizing the expected
          marginal hypervolume contribution. For ZDT1, the Pareto front is
          f1 = x1, f2 = g(x2..xn)(1 - sqrt(f1/g)). We use a proxy that
          balances exploration of x1 (which directly sets f1) with exploitation
          of low f2 (by pushing x2..xn toward the g=1 optimum).
      3.  Use a greedy replacement scheme: after each new evaluation, keep only
          the non-dominated subset, and track the best hypervolume-contributing
          points to guide future sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # --- Phase 1: Coarse stratified sampling to map the front ---
        # Use a fraction of budget for initial exploration
        n_init = max(10, min(40, budget // 5))
        # Stratify x1 uniformly, and for each x1, try a few x2..xn configurations
        # For ZDT1, g(x2..xn) = 1 + 10*(n-1) when all x_i = 0, and g=1 at the optimum
        # The Pareto front corresponds to g=1, which requires specific x2..xn values.
        # We'll sample x2..xn to try to hit low g values.

        # Generate initial population: stratified in x1, random in x2..xn
        x1_vals = np.linspace(0.0, 1.0, n_init)
        X_init = np.zeros((n_init, n))
        for i, x1v in enumerate(x1_vals):
            X_init[i, 0] = x1v
            # For x2..xn, try to push toward the g=1 optimum
            # The g=1 optimum for ZDT1 is at x_i = 0.5 + 0.5*sin(4*pi*x_i) for i>=1
            # But since we don't know the exact problem, use a heuristic:
            # Sample x2..xn with a mix of zeros (to get high g, bad) and
            # values near 0.5 (which tend to give lower g for ZDT1)
            X_init[i, 1:] = 0.5 + 0.3 * rng.standard_normal(n - 1)
            X_init[i, 1:] = np.clip(X_init[i, 1:], lo[1:], hi[1:])

        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Keep track of all evaluated points and their objectives
        all_X = X_init.copy()
        all_F = F_init.copy()

        # --- Phase 2: Adaptive refinement via marginal hypervolume gain ---
        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        def compute_hv(F_set):
            """Compute exact 2D hypervolume for a set of points vs reference."""
            if len(F_set) == 0:
                return 0.0
            # Only count points strictly better than ref
            mask = np.all(F_set < ref, axis=1)
            F_set = F_set[mask]
            if len(F_set) == 0:
                return 0.0
            # Sort by f1
            order = np.argsort(F_set[:, 0])
            F_set = F_set[order]
            hv = 0.0
            prev_f2 = ref[1]
            for f in F_set:
                if f[0] >= ref[0]:
                    break
                if f[1] < prev_f2:
                    hv += (prev_f2 - f[1]) * (ref[0] - f[0])
                    prev_f2 = f[1]
            return hv

        def nondominated_mask(F_set):
            """Return boolean mask of non-dominated points."""
            m = len(F_set)
            mask = np.ones(m, dtype=bool)
            for i in range(m):
                if not mask[i]:
                    continue
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        mask[i] = False
                        break
            return mask

        def marginal_gain(new_point, F_set):
            """Estimate the marginal hypervolume gain of adding new_point to F_set."""
            F_augmented = np.vstack([F_set, new_point.reshape(1, -1)])
            hv_before = compute_hv(F_set)
            hv_after = compute_hv(F_augmented)
            return hv_after - hv_before

        # Get current non-dominated set
        nd_mask = nondominated_mask(all_F)
        nd_X = all_X[nd_mask]
        nd_F = all_F[nd_mask]

        # Adaptive loop
        while used < budget:
            # Generate candidate points
            # Strategy: propose points that either:
            #   a) Interpolate/extrapolate in x1 between existing nd points
            #   b) Improve f2 by adjusting x2..xn for a given x1
            #   c) Random exploration

            n_candidates = min(8, budget - used)
            candidates = []

            # (a) Interpolate x1 values between consecutive nd points
            if len(nd_F) >= 2:
                f1_sorted = np.sort(nd_F[:, 0])
                for i in range(len(f1_sorted) - 1):
                    mid_x1 = (f1_sorted[i] + f1_sorted[i + 1]) / 2.0
                    if 0.0 <= mid_x1 <= 1.0:
                        x_cand = np.zeros(n)
                        x_cand[0] = mid_x1
                        # Use the x2..xn from the closer nd point
                        idx1 = np.argmin(np.abs(nd_F[:, 0] - f1_sorted[i]))
                        idx2 = np.argmin(np.abs(nd_F[:, 0] - f1_sorted[i + 1]))
                        # Blend x2..xn
                        x_cand[1:] = 0.5 * nd_X[idx1, 1:] + 0.5 * nd_X[idx2, 1:]
                        x_cand = np.clip(x_cand, lo, hi)
                        candidates.append(x_cand)

            # (b) For each nd point, try to improve f2 by perturbing x2..xn
            for i in range(len(nd_X)):
                x_cand = nd_X[i].copy()
                # Perturb x2..xn toward lower g (for ZDT1, this means toward specific values)
                perturbation = 0.05 * rng.standard_normal(n - 1)
                x_cand[1:] += perturbation
                x_cand = np.clip(x_cand, lo, hi)
                candidates.append(x_cand)

            # (c) Random exploration
            for _ in range(2):
                x_cand = lo + (hi - lo) * rng.random(n)
                candidates.append(x_cand)

            # Evaluate candidates and pick the one with highest marginal gain
            best_gain = -np.inf
            best_cand = None
            best_fcand = None

            # Limit evaluations to remaining budget
            n_eval = min(len(candidates), budget - used)
            for x_cand in candidates[:n_eval]:
                fc = problem(x_cand)
                used += 1
                gain = marginal_gain(fc, all_F)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = x_cand
                    best_fcand = fc

            if best_cand is not None:
                all_X = np.vstack([all_X, best_cand.reshape(1, -1)])
                all_F = np.vstack([all_F, best_fcand.reshape(1, -1)])

                # Update non-dominated set
                nd_mask = nondominated_mask(all_F)
                nd_X = all_X[nd_mask]
                nd_F = all_F[nd_mask]

        return all_X
# EVOLVE-BLOCK-END
