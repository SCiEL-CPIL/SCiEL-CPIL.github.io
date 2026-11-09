"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using MO-DE with archive maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # LHS initialization for diversity
        def lhs_sample(n_samples, dim, lo, hi, rng):
            indices = np.array([rng.permutation(n_samples) for _ in range(dim)]).T
            u = (indices + rng.random((n_samples, dim))) / n_samples
            return lo + (hi - lo) * u

        # Phase 1: LHS initialization
        n_init = 30
        X_init = lhs_sample(n_init, dim, lo, hi, rng)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Archive: list of (x, f) tuples, only non-dominated
        archive = []
        for i in range(n_init):
            archive.append((X_init[i].copy(), F_init[i].copy()))

        def is_dominated_by_archive(f_new, archive):
            """Check if f_new is dominated by any point in archive."""
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def dominates_archive(f_new, archive):
            """Return indices of archive points dominated by f_new."""
            dominated = []
            for i, (_, f_old) in enumerate(archive):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    dominated.append(i)
            return dominated

        def update_archive(archive, x_new, f_new):
            """Update archive with new point, removing dominated points."""
            # Check if new point is dominated
            if is_dominated_by_archive(f_new, archive):
                return archive

            # Remove points dominated by new point
            dominated = dominates_archive(f_new, archive)
            new_archive = [p for i, p in enumerate(archive) if i not in dominated]
            new_archive.append((x_new.copy(), f_new.copy()))
            return new_archive

        # Phase 2: MO-DE main loop
        pop_size = 20
        F_factor = 0.8  # DE mutation factor
        CR = 0.9  # crossover rate

        # Initialize population from LHS + archive
        pop = []
        fit = []
        for i in range(min(n_init, pop_size)):
            pop.append(X_init[i].copy())
            fit.append(F_init[i].copy())

        # Pad population with random points if needed
        while len(pop) < pop_size:
            x_rand = lo + rng.random(dim) * span
            f_rand = problem(x_rand)
            used += 1
            pop.append(x_rand.copy())
            fit.append(f_rand.copy())

        n_de = budget - used
        for k in range(n_de):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Select parents
            if len(archive) >= 3:
                # Prefer archive points for parent selection
                a_idx = rng.integers(len(archive))
                b_idx = rng.integers(len(archive))
                c_idx = rng.integers(len(archive))
                a = archive[a_idx][0]
                b = archive[b_idx][0]
                c = archive[c_idx][0]
            else:
                # Select from population
                a_idx = rng.integers(pop_size)
                b_idx = rng.integers(pop_size)
                c_idx = rng.integers(pop_size)
                while b_idx == a_idx:
                    b_idx = rng.integers(pop_size)
                while c_idx == a_idx or c_idx == b_idx:
                    c_idx = rng.integers(pop_size)
                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]

            # DE/rand/1 mutation
            v = a + F_factor * (b - c)
            v = np.clip(v, lo, hi)

            # Binomial crossover
            j_rand = rng.integers(dim)
            trial = v.copy()
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[a_idx][j]

            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            archive = update_archive(archive, trial, f_trial)

            # Update population: replace worst member
            # Simple strategy: replace a random member if trial is non-dominated
            is_dom_pop = False
            for i in range(pop_size):
                if np.all(fit[i] <= f_trial) and np.any(fit[i] < f_trial):
                    is_dom_pop = True
                    break

            if not is_dom_pop:
                replace_idx = rng.integers(pop_size)
                pop[replace_idx] = trial.copy()
                fit[replace_idx] = f_trial.copy()

        return None
# EVOLVE-BLOCK-END
