"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution-style multi-objective optimizer with archive-guided search and polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 320.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # --- Initialization: mix structured and random points ---
        pop_size = min(40, max(15, budget // 5))

        # Generate corners (2^5 = 32)
        corners = []
        for i in range(2 ** dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(corner)

        # Structured midpoints
        structured_extra = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
            lo + 0.1 * span,
            lo + 0.9 * span,
        ]

        init_points = []
        # Add corners (sample if too many)
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_idx = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_idx])

        # Add structured extras
        for sp in structured_extra:
            if len(init_points) < pop_size:
                init_points.append(sp)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Archive management ---
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any archive point
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove archive points dominated by f
            new_ax = []
            new_af = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_ax.append(ax)
                    new_af.append(af)
            new_ax.append(x.copy())
            new_af.append(f.copy())
            archive_X = new_ax
            archive_F = new_af

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Main DE loop ---
        CR = 0.7
        F_mut = 0.5

        # Reserve budget for polish phase
        polish_budget = max(20, budget // 10)
        main_budget = budget - polish_budget

        while used < main_budget:
            if time.time() - start_time > DEADLINE:
                break

            # Select base: prefer archive if available and large enough
            if len(archive_X) >= 3 and rng.random() < 0.6:
                base_idx = rng.integers(len(archive_X))
                base_x = archive_X[base_idx]
            else:
                base_idx = rng.integers(pop_size)
                base_x = X[base_idx]

            # Select two other parents
            if len(archive_X) >= 3:
                other_idx = rng.choice(len(archive_X), size=2, replace=False)
                x1 = archive_X[other_idx[0]]
                x2 = archive_X[other_idx[1]]
            else:
                other_idx = rng.choice(pop_size, size=2, replace=False)
                x1 = X[other_idx[0]]
                x2 = X[other_idx[1]]

            # Mutation
            mutant = base_x + F_mut * (x1 - x2)
            mutant = np.clip(mutant, lo, hi)

            # Binomial crossover
            trial = base_x.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Update archive
            update_archive(trial, f_trial)

            # Population replacement
            if base_idx < pop_size:
                base_f = F[base_idx]
                # Replace if trial dominates base
                if np.all(f_trial <= base_f) and np.any(f_trial < base_f):
                    X[base_idx] = trial
                    F[base_idx] = f_trial
                # Replace if base dominates trial? No, keep base.
                # If non-dominated, replace with some probability or if better sum
                elif not (np.all(base_f <= f_trial) and np.any(base_f < f_trial)):
                    # Non-dominated: use a simple criterion
                    if np.sum(f_trial) < np.sum(base_f) or rng.random() < 0.4:
                        X[base_idx] = trial
                        F[base_idx] = f_trial

            # Periodic diversity injection
            if used % 15 == 0 and used < main_budget:
                if time.time() - start_time > DEADLINE:
                    break
                rand_x = lo + span * rng.random(dim)
                rand_f = np.array(problem(rand_x))
                used += 1
                update_archive(rand_x, rand_f)
                # Replace worst in population by sum
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # --- Polish phase: local search around archive points ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

            # Sort archive by sum of objectives, take top few
            sums = np.array([np.sum(af) for af in archive_F])
            top_idx = np.argsort(sums)[:min(10, len(archive_X))]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Small perturbations
                for _ in range(min(5, budget - used)):
                    if time.time() - start_time > DEADLINE:
                        break
                    # Gaussian perturbation with shrinking sigma
                    sigma = 0.1 * span
                    perturbed = center + rng.normal(0, sigma, dim)
                    perturbed = np.clip(perturbed, lo, hi)

                    f_pert = np.array(problem(perturbed))
                    used += 1

                    # Accept if better in at least one objective and not worse in all
                    if np.any(f_pert < center_f) and not np.all(center_f < f_pert):
                        # Check if it's truly non-dominated vs center
                        if not (np.all(center_f <= f_pert) and np.any(center_f < f_pert)):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert

        return X
# EVOLVE-BLOCK-END
