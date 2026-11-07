"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using archive-based DE with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use the global numpy random state (seeded by evaluator)
        def rand_uniform():
            return np.random.uniform(0, 1, size=self.dim)

        def rand_normal():
            return np.random.standard_normal(self.dim)

        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        evaluations = 0

        def add_to_archive(x, f):
            nonlocal evaluations
            # Check if f is dominated by any point in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove points dominated by f
            i = 0
            while i < len(archive_F):
                if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                    archive_X.pop(i)
                    archive_F.pop(i)
                else:
                    i += 1
            archive_X.append(x)
            archive_F.append(f)
            return True

        def evaluate(x):
            nonlocal evaluations
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            evaluations += 1
            add_to_archive(x.copy(), f.copy())
            return x, f

        # Initial population
        pop_size = min(20, self.budget // 4)
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations += pop_size

        for x, f in zip(X, F):
            add_to_archive(x.copy(), f.copy())

        # DE parameters
        F_de = 0.8  # Mutation factor
        CR = 0.9    # Crossover rate

        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated points for parent selection
            nd_indices = []
            for i in range(len(archive_F)):
                dominated = False
                for j in range(len(archive_F)):
                    if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)

            if not nd_indices:
                nd_indices = list(range(len(archive_F)))

            # Select parents from non-dominated set
            n_nd = len(nd_indices)
            p1_idx = nd_indices[np.random.randint(n_nd)]
            p2_idx = nd_indices[np.random.randint(n_nd)]
            p3_idx = nd_indices[np.random.randint(n_nd)]

            # DE mutation
            v = archive_X[p1_idx] + F_de * (archive_X[p2_idx] - archive_X[p3_idx])

            # Binomial crossover
            trial = archive_X[p1_idx].copy()
            jrand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.random() < CR or j == jrand:
                    trial[j] = v[j]

            trial = np.clip(trial, lo, hi)
            f_trial = np.array(problem(trial))
            evaluations += 1
            add_to_archive(trial.copy(), f_trial.copy())

            # Local polish on the best archive point occasionally
            if evaluations % 10 == 0 and len(archive_F) > 0 and time.time() < deadline - 5:
                # Pick a random non-dominated point for local search
                if nd_indices:
                    idx = nd_indices[np.random.randint(len(nd_indices))]
                    x_cur = archive_X[idx].copy()
                    f_cur = archive_F[idx].copy()

                    # Adaptive Nelder-Mead style local search
                    n_iter = min(50, (self.budget - evaluations) // 2)
                    step_size = 0.1
                    for it in range(n_iter):
                        if evaluations >= self.budget or time.time() > deadline:
                            break
                        # Try small perturbations
                        improved = False
                        for _ in range(self.dim):
                            dx = np.zeros(self.dim)
                            dx[np.random.randint(self.dim)] = step_size * (2 * np.random.random() - 1)
                            x_new = np.clip(x_cur + dx, lo, hi)
                            f_new = np.array(problem(x_new))
                            evaluations += 1
                            # Check if f_new is better than f_cur in at least one objective and not worse in others
                            if np.all(f_new <= f_cur) and np.any(f_new < f_cur):
                                x_cur = x_new
                                f_cur = f_new
                                improved = True
                                add_to_archive(x_cur.copy(), f_cur.copy())
                                break
                        if not improved:
                            step_size *= 0.5
                            if step_size < 1e-4:
                                break

        return archive_X
# EVOLVE-BLOCK-END
