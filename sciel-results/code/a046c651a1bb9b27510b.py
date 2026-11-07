"""Multi-objective optimizer for WFG4 using analytical Pareto front sampling with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical Pareto front construction with local refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        # Track all evaluated points and their objective values
        all_points = []
        all_fvals = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            return f

        # WFG4 specific: the Pareto front is determined by x_1 (shape variable)
        # The remaining variables should be set to specific values to achieve the front.
        # For WFG4 with 12 variables:
        # - x_1 controls the shape (typically in [0, 0.5] or similar)
        # - x_2 through x_12 should be set to achieve the minimum of each sub-function
        #
        # WFG4 definition (simplified for 12D):
        # The problem has two groups of variables:
        # Group 1: x_1 (shape variable)
        # Group 2: x_2 to x_12 (11 variables)
        #
        # The Pareto-optimal values for x_2...x_12 are typically:
        # For WFG4, the optimal values for the "position" variables depend on the specific
        # formulation. A common approach is to set them to values that minimize the
        # corresponding sub-objectives.
        #
        # For a standard WFG4 with M=3 objectives and K=12 variables:
        # The shape variable is x_1, and the remaining K-1 = 11 variables are position variables.
        # The Pareto front is achieved when:
        # - x_1 varies in its range
        # - x_2...x_12 are set to specific constants
        #
        # For WFG4, the typical optimal values for position variables are:
        # x_i = 0.5 * (hi_i - lo_i) + lo_i for i >= 2 (midpoint) or specific values
        #
        # Let's use a more robust approach: sample x_1 uniformly and use local search
        # to find the best values for x_2...x_12 for each x_1 value.

        # Strategy:
        # 1. Sample x_1 uniformly across its range (this defines the Pareto front shape)
        # 2. For each x_1 value, use a small local search to find optimal x_2...x_12
        # 3. Use remaining budget for additional exploration

        # Determine the range of x_1 (shape variable)
        # For WFG4, x_1 typically ranges over [0, 0.5] or the full bound range
        # We'll sample x_1 across its full bound range

        n_front_samples = 80  # number of points to sample along the Pareto front
        remaining_budget = budget - n_front_samples

        if remaining_budget < 20:
            n_front_samples = max(10, budget - 20)
            remaining_budget = budget - n_front_samples

        # Sample x_1 values uniformly
        x1_range = hi[0] - lo[0]
        x1_values = np.linspace(lo[0], hi[0], n_front_samples)

        # For each x_1 value, find good values for x_2...x_12
        # Use a simple approach: try several candidate values and keep the best
        # For WFG4, the optimal position variables are often at specific values

        # Common optimal values for WFG4 position variables:
        # Often x_i = 0.5 for i >= 2 (in normalized space)
        # Or specific values derived from the problem structure

        # Let's try multiple candidate patterns for x_2...x_12
        candidate_patterns = []

        # Pattern 1: All midpoints
        mid = (lo + hi) / 2.0
        candidate_patterns.append(mid.copy())

        # Pattern 2: All at lower bound
        candidate_patterns.append(lo.copy())

        # Pattern 3: All at upper bound
        candidate_patterns.append(hi.copy())

        # Pattern 4: Alternating low/high
        alt = np.where(np.arange(dim) % 2 == 0, lo, hi)
        candidate_patterns.append(alt)

        # For each x_1 value, evaluate a few candidate patterns and keep the best
        # "best" in terms of being non-dominated or having good HV contribution

        def hv_contribution_simple(f, archive):
            """Compute approximate HV contribution of point f to archive."""
            if np.any(f >= ref):
                return 0.0
            # Check if dominated
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            # Box volume
            vol = 1.0
            for i in range(n_obj):
                d = ref[i] - f[i]
                if d <= 0:
                    return 0.0
                vol *= d
            return vol

        # For each x_1 sample, find the best configuration
        for x1_val in x1_values:
            if eval_count >= budget:
                break
            if time.time() > deadline - 10.0:
                break

            best_f = None
            best_x = None
            best_score = -1.0

            # Try each candidate pattern
            for pattern in candidate_patterns:
                if eval_count >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break

                x = pattern.copy()
                x[0] = x1_val
                x = np.clip(x, lo, hi)

                f = evaluate(x)

                # Compute score (simple HV contribution)
                score = hv_contribution_simple(f, all_fvals)

                if score > best_score:
                    best_score = score
                    best_f = f.copy()
                    best_x = x.copy()

            if best_x is not None:
                all_points.append(best_x)
                all_fvals.append(best_f)

        # Use remaining budget for local refinement
        # Pick a subset of the best points and refine them
        if eval_count < budget and time.time() < deadline - 10.0:
            # Get non-dominated points
            def get_non_dominated(fvals):
                n = len(fvals)
                if n == 0:
                    return []
                nd = list(range(n))
                for i in range(n):
                    if i not in nd:
                        continue
                    for j in range(n):
                        if i == j:
                            continue
                        if j not in nd:
                            continue
                        fi = fvals[i]
                        fj = fvals[j]
                        if np.all(fj <= fi) and np.any(fj < fi):
                            nd.remove(i)
                            break
                return nd

            nd_idx = get_non_dominated(all_fvals)
            if nd_idx:
                # Refine each non-dominated point with small perturbations
                rng = np.random.default_rng(123)
                for idx in nd_idx:
                    if eval_count >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break

                    base_x = all_points[idx].copy()
                    base_f = all_fvals[idx].copy()

                    # Try small perturbations in each dimension
                    for d in range(dim):
                        if eval_count >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break

                        span = hi[d] - lo[d]
                        for step_frac in [0.05, 0.02, 0.01]:
                            for direction in [-1, 1]:
                                if eval_count >= budget:
                                    break
                                if time.time() > deadline - 5.0:
                                    break

                                new_x = base_x.copy()
                                new_x[d] = base_x[d] + direction * step_frac * span
                                new_x[d] = np.clip(new_x[d], lo[d], hi[d])

                                if np.allclose(new_x, base_x):
                                    continue

                                new_f = evaluate(new_x)

                                # Check if this is non-dominated w.r.t. current archive
                                is_dominated = False
                                for af in all_fvals:
                                    if np.all(af <= new_f) and np.any(af < new_f):
                                        is_dominated = True
                                        break

                                if not is_dominated:
                                    # Check if it dominates any existing points
                                    dominates_any = False
                                    for j, af in enumerate(all_fvals):
                                        if np.all(new_f <= af) and np.any(new_f < af):
                                            dominates_any = True
                                            break

                                    # Add if non-dominated
                                    all_points.append(new_x)
                                    all_fvals.append(new_f)

                                    # Remove dominated points
                                    if dominates_any:
                                        new_nd = get_non_dominated(all_fvals)
                                        all_points = [all_points[i] for i in new_nd]
                                        all_fvals = [all_fvals[i] for i in new_nd]

        return all_points
# EVOLVE-BLOCK-END
