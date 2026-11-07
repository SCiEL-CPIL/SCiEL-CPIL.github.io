"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using archive-based adaptive local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track all evaluated points
        all_F = []
        all_X = []
        used = 0

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            all_F.append(fc)
            all_X.append(x.copy())
            used += 1
            return fc

        # Archive: list of (x, f) for non-dominated points
        archive_X = []
        archive_F = []

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F_list):
            for af in archive_F_list:
                if dominates(af, f):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by existing archive
            for af in archive_F:
                if dominates(af, f):
                    return
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Initialize: spread x1 across its range (WFG4's key variable), randomize others
        n_init = 20
        X_init = np.zeros((n_init, dim))
        X_init[:, 0] = np.linspace(lo[0], hi[0], n_init)
        X_init[:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random((n_init, dim - 1))

        for x in X_init:
            if used >= budget or time.time() > deadline:
                break
            f = evaluate(x)
            add_to_archive(x, f)

        # Adaptive local search phase
        step_sizes = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005]
        current_step_idx = 0
        no_improve_count = 0
        max_no_improve = 15

        while used < budget and time.time() < deadline:
            # Select parent from archive (random or best)
            if len(archive_F) > 0:
                if rng.random() < 0.7:
                    # Pick from archive, biased towards better (lower sum) points
                    sums = np.array([np.sum(f) for f in archive_F])
                    # Soft-min weighting: lower sum -> higher probability
                    weights = np.exp(-sums / (0.5 * np.std(sums) + 1e-10))
                    weights /= weights.sum()
                    parent_idx = rng.choice(len(archive_F), p=weights)
                else:
                    parent_idx = rng.integers(len(archive_F))
                parent_x = archive_X[parent_idx].copy()
            else:
                parent_x = lo + (hi - lo) * rng.random(dim)

            # Generate offspring
            step_size = step_sizes[min(current_step_idx, len(step_sizes) - 1)]
            # Adaptive: if stuck, reduce step; if improving, keep or increase
            if no_improve_count >= max_no_improve:
                if current_step_idx < len(step_sizes) - 1:
                    current_step_idx += 1
                no_improve_count = 0
            elif no_improve_count == 0 and current_step_idx > 0:
                current_step_idx -= 1

            # Gaussian perturbation
            offset = rng.normal(0, step_size, dim)
            child_x = parent_x + offset
            child_x = np.clip(child_x, lo, hi)

            # Also try a few deterministic perturbations along axes
            n_trials = 3
            for t in range(n_trials):
                if used >= budget or time.time() > deadline:
                    break

                if t == 0:
                    trial_x = child_x
                elif t == 1:
                    # Perturb only x1 (the key WFG4 variable)
                    trial_x = parent_x.copy()
                    trial_x[0] = parent_x[0] + rng.normal(0, step_size * 2)
                    trial_x[0] = np.clip(trial_x[0], lo[0], hi[0])
                else:
                    # Perturb a random subset of variables
                    trial_x = parent_x.copy()
                    n_vars = max(1, dim // 3)
                    var_idx = rng.choice(dim, size=n_vars, replace=False)
                    trial_x[var_idx] = parent_x[var_idx] + rng.normal(0, step_size, n_vars)
                    trial_x[var_idx] = np.clip(trial_x[var_idx], lo[var_idx], hi[var_idx])

                f = evaluate(trial_x)

                # Check if non-dominated by archive
                if not is_dominated(f, archive_F):
                    # Remove dominated points from archive
                    new_archive_X = []
                    new_archive_F = []
                    for i in range(len(archive_F)):
                        if not dominates(f, archive_F[i]):
                            new_archive_X.append(archive_X[i])
                            new_archive_F.append(archive_F[i])
                    new_archive_X.append(trial_x.copy())
                    new_archive_F.append(f.copy())
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # Periodically add random exploration
            if used < budget and rng.random() < 0.05:
                if time.time() > deadline:
                    break
                rand_x = lo + (hi - lo) * rng.random(dim)
                f = evaluate(rand_x)
                if not is_dominated(f, archive_F):
                    new_archive_X = []
                    new_archive_F = []
                    for i in range(len(archive_F)):
                        if not dominates(f, archive_F[i]):
                            new_archive_X.append(archive_X[i])
                            new_archive_F.append(archive_F[i])
                    new_archive_X.append(rand_x.copy())
                    new_archive_F.append(f.copy())
                    archive_X = new_archive_X
                    archive_F = new_archive_F

        return all_F
# EVOLVE-BLOCK-END
