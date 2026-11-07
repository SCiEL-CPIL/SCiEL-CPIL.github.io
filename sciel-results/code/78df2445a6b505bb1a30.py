"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer with deadline guard, NSGA-II-style search, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def dominates(a, b):
            """True if a dominates b (all <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f."""
            for af in archive_f:
                if dominates(af, f):
                    return True
            return False

        def non_dominated_indices(F):
            """Return indices of non-dominated points in F (array of shape (n, n_obj))."""
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Initialize population
        pop_size = min(40, budget // 3)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of all non-dominated points seen so far
        archive_X = [X[i] for i in range(len(X))]
        archive_F = [F[i] for i in range(len(F))]

        # Update archive
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            if not is_dominated(f, archive_F):
                # Remove any archive points dominated by f
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not dominates(f, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F

        for i in range(len(archive_X)):
            pass  # already set

        # Main loop
        generation = 0
        while used < budget:
            if time.time() > deadline - 30:
                break

            # Compute non-dominated set in current population
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = list(range(len(F)))

            # Select parents from non-dominated set
            n_parents = max(2, len(nd_idx) // 2)
            parent_indices = rng.choice(nd_idx, size=min(n_parents, len(nd_idx)), replace=False)
            p1_idx = parent_indices[0]
            p2_idx = parent_indices[1] if len(parent_indices) > 1 else parent_indices[0]

            # Crossover (SBX-like or simple blend)
            if rng.random() < 0.7:
                # Blend crossover
                alpha = rng.random()
                child = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
            else:
                # Single-point crossover
                cross_point = rng.integers(1, dim)
                child = np.concatenate([X[p1_idx][:cross_point], X[p2_idx][cross_point:]])

            # Adaptive mutation: step size decreases as budget depletes
            progress = used / budget
            step_scale = 0.3 * (1.0 - 0.7 * progress)
            mutation = step_scale * span * rng.standard_normal(dim)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use a simple crowding-like metric: sum of objectives normalized
            # Or just replace the last-added point that is dominated
            worst_idx = -1
            for i in range(len(F)):
                if is_dominated(F[i], F):
                    worst_idx = i
                    break
            if worst_idx == -1:
                # Replace the point with the worst sum
                worst_idx = int(np.argmax(np.sum(F, axis=1)))

            X[worst_idx] = child
            F[worst_idx] = fc

            update_archive(child, fc)

            # Periodically do local refinement on best archive points
            if used % 20 == 0 and time.time() < deadline - 60:
                # Pick a few promising archive points and do local search
                n_refine = min(3, len(archive_X))
                if n_refine > 0:
                    # Pick points that are on the edges of the front
                    idxs = rng.choice(len(archive_X), size=n_refine, replace=False)
                    for idx in idxs:
                        if time.time() > deadline - 30:
                            break
                        start_x = archive_X[idx].copy()
                        try:
                            # Local Nelder-Mead on a scalarized objective
                            weights = rng.random(n_obj)
                            weights /= weights.sum()
                            def scalar_obj(x):
                                fx = problem(x)
                                return float(np.dot(weights, fx))
                            res = minimize(scalar_obj, start_x, method='Nelder-Mead',
                                         options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                            if res.fun < np.dot(weights, problem(start_x)) - 1e-8:
                                fx = problem(res.x)
                                used += 1
                                update_archive(res.x, fx)
                        except Exception:
                            pass

        return X
# EVOLVE-BLOCK-END
