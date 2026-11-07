"""Multi-objective optimizer for RE37: Enhanced DE with adaptive archive and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Enhanced multi-objective optimizer using adaptive DE with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if len(archive_F) == 0:
                return False
            return np.any(np.all(archive_F <= f, axis=1) & np.any(archive_F < f, axis=1))

        def update_archive(new_x, new_f, archive_X, archive_F):
            """Add new point to archive if non-dominated, remove dominated archive points."""
            if is_dominated(new_f, archive_F):
                return archive_X, archive_F

            # Remove archive points dominated by new point
            mask = ~(np.all(archive_F >= new_f, axis=1) & np.any(archive_F > new_f, axis=1))
            keep_idx = np.where(mask)[0]

            new_archive_X = np.vstack([archive_X[keep_idx], new_x.reshape(1, -1)])
            new_archive_F = np.vstack([archive_F[keep_idx], new_f.reshape(1, -1)])
            return new_archive_X, new_archive_F

        # Initial population: stratified sampling for better coverage
        np.random.seed(42)
        pop_size = min(30, budget // 3)

        # Use Latin Hypercube-like sampling for initial population
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Stratified sampling along each dimension
            strata = np.linspace(0, 1, pop_size + 1)
            offsets = np.random.uniform(0, 1, pop_size)
            X[:, d] = (strata[:pop_size] + offsets) / pop_size
            X[:, d] = lo[d] + (hi[d] - lo[d]) * X[:, d]

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Initialize archive
        archive_X = X.copy()
        archive_F = F.copy()

        # Differential Evolution parameters
        Fp = 0.8  # mutation factor
        Cr = 0.8  # crossover rate

        # Local search parameters
        ls_radius = 0.05
        ls_iterations = 3

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Adaptive mutation factor based on archive diversity
            if len(archive_F) > 1:
                spread = np.max(archive_F, axis=0) - np.min(archive_F, axis=0)
                avg_spread = np.mean(spread)
                Fp = np.clip(0.5 + 0.5 * avg_spread, 0.5, 1.0)

            # Generate multiple trial vectors per iteration for efficiency
            n_trials = min(3, budget - used)

            for _ in range(n_trials):
                if used >= budget or time.time() > deadline:
                    break

                # DE/rand/1 mutation
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                mutant = X[a] + Fp * (X[b] - X[c])
                mutant = np.clip(mutant, lo, hi)

                # Binary crossover
                base_idx = np.random.randint(pop_size)
                trial = X[base_idx].copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < Cr or j == jrand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Update archive
                archive_X, archive_F = update_archive(trial, f_trial, archive_X, archive_F)

                # Local search around promising points
                if len(archive_F) > 0 and np.random.random() < 0.3:
                    # Pick a random archive point for local search
                    ls_idx = np.random.randint(len(archive_F))
                    ls_center = archive_X[ls_idx]

                    for ls_iter in range(ls_iterations):
                        if used >= budget or time.time() > deadline:
                            break

                        # Generate perturbation
                        direction = np.random.randn(dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        ls_trial = ls_center + ls_radius * direction
                        ls_trial = np.clip(ls_trial, lo, hi)

                        f_ls = problem(ls_trial)
                        used += 1

                        archive_X, archive_F = update_archive(ls_trial, f_ls, archive_X, archive_F)

                        # Update local center if improved
                        if not is_dominated(f_ls, archive_F):
                            ls_center = ls_trial

                # Update population: replace worst if trial is better
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(f_trial) < np.sum(F[worst_idx]):
                    X[worst_idx] = trial
                    F[worst_idx] = f_trial
                elif np.random.random() < 0.15:
                    # Diversity maintenance
                    X[worst_idx] = trial
                    F[worst_idx] = f_trial

            # Periodic population refresh to maintain diversity
            if used % 50 < n_trials and used < budget:
                n_refresh = min(5, budget - used)
                for _ in range(n_refresh):
                    if used >= budget or time.time() > deadline:
                        break
                    new_x = lo + (hi - lo) * np.random.random(dim)
                    new_f = problem(new_x)
                    used += 1
                    archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)

                    # Replace a random population member
                    repl_idx = np.random.randint(pop_size)
                    X[repl_idx] = new_x
                    F[repl_idx] = new_f

        return archive_X

# EVOLVE-BLOCK-END
