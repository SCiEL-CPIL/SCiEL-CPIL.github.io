"""Multi-start Nelder-Mead ensemble with Latin Hypercube initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s wall-clock limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, f0, max_iter):
            """Run Nelder-Mead starting from x0 with value f0. Returns (best_x, best_f, calls_used)."""
            nonlocal calls
            local_calls = 0
            alpha = 0.3  # Initial simplex size

            simplex = [x0.copy()]
            f_simplex = [f0]

            # Build simplex
            for i in range(dim):
                if not time_ok() or not budget_ok():
                    return x0, f0, local_calls
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                local_calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)

            if len(simplex) < 2:
                return x0, f0, local_calls

            # Nelder-Mead parameters
            gamma = 1.0
            rho = 2.0
            sigma_nm = 0.5
            shrink = 0.5

            for _ in range(max_iter):
                if not time_ok() or not budget_ok():
                    break

                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]

                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                local_calls += 1
                record(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    local_calls += 1
                    record(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    local_calls += 1
                    record(x_c, f_c)

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            local_calls += 1
                            record(simplex[i], f_simplex[i])
                        if not time_ok() or not budget_ok():
                            break

                # Check for convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-12:
                    break

            return best_x, best_f, local_calls

        # Generate diverse starting points using Latin Hypercube Sampling
        rng = np.random.default_rng(42)
        n_starts = 5

        # Latin Hypercube: divide each dimension into n_starts equal intervals
        lhb_points = np.zeros((n_starts, dim))
        for d in range(dim):
            perm = rng.permutation(n_starts)
            for i in range(n_starts):
                # Place point in interval i, at random position within that interval
                lhb_points[i, d] = -5 + (i + rng.random()) * (10.0 / n_starts)

        # Also add a center point and a couple of random points for diversity
        extra_points = np.zeros((3, dim))
        extra_points[0] = 0.0  # Center
        extra_points[1] = rng.uniform(-4, 4, dim)
        extra_points[2] = rng.uniform(-4, 4, dim)

        all_starts = np.vstack([lhb_points, extra_points])
        n_starts_total = len(all_starts)

        # Evaluate all starting points
        start_values = np.zeros(n_starts_total)
        for i in range(n_starts_total):
            if not time_ok() or not budget_ok():
                break
            x_i = all_starts[i]
            f_i = func(x_i)
            calls += 1
            start_values[i] = f_i
            record(x_i, f_i)

        # Sort starts by function value
        sorted_indices = np.argsort(start_values)

        # Allocate remaining budget among starts, prioritizing better ones
        remaining = budget - calls
        if remaining <= 0 or not time_ok():
            return best_x, best_f

        # Distribute budget: give more iterations to better starts
        # Use a simple allocation: proportional to rank quality
        total_weight = 0
        weights = np.zeros(n_starts_total)
        for rank, idx in enumerate(sorted_indices):
            # Better starts (lower rank) get more weight
            weights[idx] = 1.0 / (rank + 1)
            total_weight += weights[idx]

        # Normalize weights and allocate iterations
        iters_per_start = np.zeros(n_starts_total, dtype=int)
        for idx in sorted_indices:
            iters_per_start[idx] = max(10, int(remaining * weights[idx] / total_weight))

        # Cap total iterations to not exceed remaining budget
        total_iters = np.sum(iters_per_start)
        if total_iters > remaining:
            scale = remaining / total_iters
            for idx in range(n_starts_total):
                iters_per_start[idx] = max(5, int(iters_per_start[idx] * scale))

        # Run Nelder-Mead from each start, in order of quality
        for idx in sorted_indices:
            if not time_ok() or not budget_ok():
                break
            if iters_per_start[idx] <= 0:
                continue
            x0 = all_starts[idx].copy()
            f0 = start_values[idx]
            nelder_mead(x0, f0, iters_per_start[idx])

        return best_x, best_f
# EVOLVE-BLOCK-END
