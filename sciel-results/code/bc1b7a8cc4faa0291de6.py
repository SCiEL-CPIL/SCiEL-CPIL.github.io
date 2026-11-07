import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_func(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            update_best(x, f)
            return f

        # Powell's method with bounded line search (golden section)
        def powell(x0, max_iter=10):
            nonlocal calls, deadline
            x = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
            f = eval_func(x)
            if f is None:
                return x, best_f

            # Initial directions: coordinate axes
            directions = [np.eye(dim)[i] for i in range(dim)]

            for iteration in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                old_x = x.copy()
                old_f = f
                min_idx = 0
                min_delta = 0.0

                for i in range(dim):
                    if time.time() > deadline or calls >= budget:
                        break
                    # Line search along directions[i]
                    new_x, new_f = line_search_bounded(x, directions[i])
                    delta = new_f - f
                    if delta < min_delta:
                        min_delta = delta
                        min_idx = i
                    x = new_x
                    f = new_f

                # Check convergence
                if abs(min_delta) < 1e-10:
                    break

                # Update directions: replace worst with new direction
                if min_idx > 0:
                    directions[min_idx - 1] = directions[min_idx]
                directions[min_idx] = x - old_x
                # Normalize to avoid numerical issues
                norm = np.linalg.norm(directions[min_idx])
                if norm > 1e-15:
                    directions[min_idx] = directions[min_idx] / norm

            return x, f

        def line_search_bounded(x, direction, max_steps=20):
            nonlocal calls, deadline
            # Find good step size via golden section search
            # First, find a bracket
            step = 1.0
            f0 = eval_func(x)
            if f0 is None:
                return x, best_f

            # Try to find a step that reduces the function
            best_step = 0.0
            best_f = f0
            best_x = x.copy()

            # Try several step sizes
            for attempt in range(max_steps):
                if time.time() > deadline or calls >= budget:
                    break
                # Try positive and negative steps
                for sign in [1.0, -1.0]:
                    x_try = x + sign * step * direction
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = eval_func(x_try)
                    if f_try is None:
                        break
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        best_step = sign * step
                # Reduce step size
                step *= 0.5
                if step < 1e-8:
                    break

            return best_x, best_f

        # Stratified Latin Hypercube initialization
        def lhc_sample(n_points, dim, seed=42):
            rng = np.random.RandomState(seed)
            points = np.zeros((n_points, dim))
            for d in range(dim):
                # Stratify into n_points equal intervals
                intervals = np.linspace(-5.0, 5.0, n_points + 1)
                for i in range(n_points):
                    # Uniform within the interval
                    points[i, d] = rng.uniform(intervals[i], intervals[i + 1])
            # Shuffle rows to decorrelate
            perm = rng.permutation(n_points)
            return points[perm]

        # Phase 1: Global exploration with LHC + Powell
        # Use a good fraction of budget for global search
        n_global = 40  # Number of global starts
        n_points = min(n_global, max(10, budget // 15))

        # Generate LHC points
        lhc_points = lhc_sample(n_points, dim, seed=123)

        # Also add some random points for diversity
        rng = np.random.RandomState(456)
        random_points = rng.uniform(-5.0, 5.0, size=(n_points, dim))

        # Combine and deduplicate
        all_starts = np.vstack([lhc_points, random_points])

        # Sort by distance to origin (prioritize center)
        dists = np.linalg.norm(all_starts, axis=1)
        order = np.argsort(dists)
        all_starts = all_starts[order]

        # Evaluate each start with Powell
        for start_point in all_starts:
            if time.time() > deadline or calls >= budget:
                break
            # Quick evaluation of the start point
            f_start = eval_func(start_point)
            if f_start is None:
                break
            # Run Powell from this start
            powell(start_point, max_iter=8)

        # Phase 2: Local refinement around the best point
        if time.time() < deadline and calls < budget:
            # Try small perturbations around best_x
            rng2 = np.random.RandomState(789)
            n_refine = min(30, max(5, (budget - calls) // 10))
            for i in range(n_refine):
                if time.time() > deadline or calls >= budget:
                    break
                # Small perturbation
                perturbation = rng2.normal(0, 0.5, size=dim)
                x_try = best_x + perturbation
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = eval_func(x_try)
                if f_try is None:
                    break
                # Short Powell from this perturbed point
                powell(x_try, max_iter=5)

        # Phase 3: Final polish if budget remains
        if time.time() < deadline and calls < budget:
            # One more Powell from the best point
            powell(best_x, max_iter=10)

        return best_x, best_f
