"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive reset and multi-start diversification."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Wrap func to count calls and enforce budget
        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                raise RuntimeError("budget exhausted")
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Nelder-Mead simplex method
        def nelder_mead(start, max_iter, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
            """Run Nelder-Mead from start, using up to max_iter evaluations."""
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = start
            for i in range(n):
                step = 0.1 if abs(start[i]) < 1e-8 else 0.05 * abs(start[i])
                simplex[i + 1] = start.copy()
                simplex[i + 1][i] += step

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if call_count[0] >= max_calls:
                    return simplex[0].copy(), fvals[0]
                fvals[i] = counted_func(simplex[i])

            # Track best
            idx_best = np.argmin(fvals)
            update_best(simplex[idx_best], fvals[idx_best])

            for _ in range(max_iter):
                if call_count[0] >= max_calls:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function values
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = counted_func(x_ref)
                update_best(x_ref, f_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    if call_count[0] < max_calls:
                        f_exp = counted_func(x_exp)
                        update_best(x_exp, f_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    if call_count[0] < max_calls:
                        f_con = counted_func(x_con)
                        update_best(x_con, f_con)
                        if f_con <= min(f_ref, fvals[-1]):
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if call_count[0] < max_calls:
                                    fvals[i] = counted_func(simplex[i])
                                    update_best(simplex[i], fvals[i])
                    else:
                        break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

            idx_best = np.argmin(fvals)
            return simplex[idx_best].copy(), fvals[idx_best]

        # Multi-start strategy: try several initial points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points in the box
        for _ in range(7):
            starts.append(rng.uniform(-4, 4, dim))
        # Some structured points
        starts.append(np.ones(dim) * 2.0)
        starts.append(-np.ones(dim) * 2.0)
        starts.append(np.array([3.0, -3.0, 3.0, -3.0, 3.0])[:dim])
        starts.append(np.array([-3.0, 3.0, -3.0, 3.0, -3.0])[:dim])

        # Sort starts by distance from origin (prioritize center)
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Allocate budget across starts: more budget for earlier starts
        # First start gets ~40% of budget, subsequent ones get less
        total_starts = len(starts)
        # We'll run Nelder-Mead from each start, with decreasing budget
        # But we need to be careful about total budget

        # Strategy: run NM from best few starts, using remaining budget
        # First, do a quick evaluation of all starts to rank them
        start_fvals = []
        for s in starts:
            if call_count[0] >= max_calls:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_s = counted_func(s_clipped)
            start_fvals.append((f_s, s_clipped))
            update_best(s_clipped, f_s)

        # Sort by function value
        start_fvals.sort(key=lambda x: x[0])

        # Now run Nelder-Mead from the top starts
        # Allocate remaining budget
        remaining = max_calls - call_count[0]
        if remaining <= 0:
            return best_x, best_f

        # Use top 3 starts, allocate budget proportionally
        num_nm_starts = min(3, len(start_fvals))
        # Give more budget to the best start
        budgets = []
        for i in range(num_nm_starts):
            if i == 0:
                budgets.append(int(remaining * 0.5))
            elif i == 1:
                budgets.append(int(remaining * 0.3))
            else:
                budgets.append(remaining - sum(budgets))

        for i in range(num_nm_starts):
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline:
                break
            if budgets[i] <= 0:
                continue

            start_x = start_fvals[i][1]
            x_best, f_best = nelder_mead(start_x, budgets[i])
            update_best(x_best, f_best)

        # If we still have budget, do a final polish with a small NM
        remaining = max_calls - call_count[0]
        if remaining > 10 and best_x is not None:
            if time.time() < deadline:
                x_best, f_best = nelder_mead(best_x, remaining)
                update_best(x_best, f_best)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
