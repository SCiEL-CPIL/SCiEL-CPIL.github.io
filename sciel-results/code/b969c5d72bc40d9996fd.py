# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing (MOSA) optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 265  # safety margin before 300s/330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Normalize bounds for step sizing
        range_vec = hi - lo
        mid = (lo + hi) / 2.0

        # Reference point for hypervolume-like scoring
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive: list of (x, f) pairs, non-dominated
        archive_x = []
        archive_f = []

        def is_dominated(f_new, archive_f_list):
            """Check if f_new is dominated by any point in archive_f_list."""
            for f in archive_f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x_new, f_new):
            """Add to archive if not dominated; remove dominated points."""
            # Check if dominated by existing archive
            if is_dominated(f_new, archive_f):
                return False

            # Remove points dominated by f_new
            i = 0
            while i < len(archive_f):
                if dominates(f_new, archive_f[i]):
                    archive_x.pop(i)
                    archive_f.pop(i)
                else:
                    i += 1

            archive_x.append(x_new)
            archive_f.append(f_new)
            return True

        # Initialize with a diverse set of solutions
        init_size = min(30, budget)
        for _ in range(init_size):
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            add_to_archive(x.copy(), f.copy())

        used = init_size

        # Temperature schedule
        T0 = 1.0
        T_min = 1e-6
        cooling_rate = 0.995

        # Current solution: pick a random archive point
        if archive_x:
            cur_idx = rng.integers(len(archive_x))
            current_x = archive_x[cur_idx].copy()
            current_f = archive_f[cur_idx].copy()
        else:
            current_x = mid.copy()
            current_f = problem(current_x)
            add_to_archive(current_x.copy(), current_f.copy())
            used += 1

        T = T0

        # Main SA loop
        while used < budget:
            if time.time() > deadline:
                break

            # Generate neighbor: perturb current solution
            # Adaptive step size: start large, shrink over time
            step_scale = 0.1 + 0.15 * (T / T0)
            noise = rng.standard_normal(dim) * step_scale
            neighbor_x = current_x + noise * range_vec
            neighbor_x = np.clip(neighbor_x, lo, hi)

            neighbor_f = problem(neighbor_x)
            used += 1

            # Decide acceptance
            # If neighbor dominates current, accept
            if dominates(neighbor_f, current_f):
                current_x = neighbor_x.copy()
                current_f = neighbor_f.copy()
                add_to_archive(current_x.copy(), current_f.copy())
            elif not is_dominated(neighbor_f, archive_f):
                # Non-dominated: add to archive and accept
                current_x = neighbor_x.copy()
                current_f = neighbor_f.copy()
                add_to_archive(current_x.copy(), current_f.copy())
            else:
                # Dominated: accept with probability exp(-delta/T)
                # Use a scalar metric: sum of objectives (normalized)
                delta = np.sum(neighbor_f - current_f)
                if delta < 0:
                    current_x = neighbor_x.copy()
                    current_f = neighbor_f.copy()
                    add_to_archive(current_x.copy(), current_f.copy())
                else:
                    # Accept with probability exp(-delta / T)
                    if T > T_min:
                        prob = np.exp(-delta / max(T, 1e-10))
                        if rng.random() < prob:
                            current_x = neighbor_x.copy()
                            current_f = neighbor_f.copy()
                            # Don't add to archive if dominated
                    # Otherwise reject

            # Occasionally restart from a random archive point to diversify
            if len(archive_x) > 3 and rng.random() < 0.05:
                idx = rng.integers(len(archive_x))
                current_x = archive_x[idx].copy()
                current_f = archive_f[idx].copy()

            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return archive_x
# EVOLVE-BLOCK-END
