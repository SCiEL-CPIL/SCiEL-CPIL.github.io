"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using adaptive DE with hypervolume-driven archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 325.0

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        rng = np.random

        # --- Archive with hypervolume contribution tracking ---
        archive_X = []
        archive_F = []

        def _dominates(a, b):
            """True if a dominates b (a <= b and a < b in at least one dim)."""
            return np.all(a <= b) and np.any(a < b)

        def _hv_contribution(f, archive_F_list, ref):
            """Approximate hypervolume contribution of f given archive points."""
            # Quick check: is f dominated by any archive point?
            for af in archive_F_list:
                if _dominates(af, f):
                    return 0.0
            # Simple 3D hypervolume contribution approximation
            # Only count if f is better than ref in all dims
            if np.any(f >= ref):
                return 0.0
            # Approximate: volume of box from f to ref, minus overlap with archive
            vol = np.prod(np.maximum(ref - f, 0))
            # Subtract overlap with archive points that are "closer" to ref
            overlap = 0.0
            for af in archive_F_list:
                if np.any(af >= ref):
                    continue
                # Overlap box: from max(f, af) to ref
                lo_box = np.maximum(f, af)
                if np.all(lo_box < ref):
                    overlap += np.prod(ref - lo_box)
            return max(0.0, vol - overlap)

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any archive point
            for af in archive_F:
                if _dominates(af, f):
                    return
            # Remove archive points dominated by f
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not _dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            # Cap archive size by removing lowest HV contribution
            max_archive = 60
            if len(new_F) > max_archive:
                ref = np.array([1705.0, 12.0, 0.3])
                contributions = []
                for i, f_i in enumerate(new_F):
                    others = [new_F[j] for j in range(len(new_F)) if j != i]
                    c = _hv_contribution(f_i, others, ref)
                    contributions.append(c)
                # Remove point with lowest contribution
                min_idx = int(np.argmin(contributions))
                new_X.pop(min_idx)
                new_F.pop(min_idx)
            archive_X = new_X
            archive_F = new_F

        # --- Initialization: diverse structured sampling ---
        pop_size = 40
        init_points = []

        # Center
        init_points.append(mid.copy())

        # Axis-aligned points (variations along each dimension)
        for d in range(dim):
            for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
                p = mid.copy()
                p[d] = lo[d] + frac * span[d]
                init_points.append(p.copy())

        # Random fill
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Main loop: adaptive DE with archive-guided selection ---
        # Phase 1: exploration (first 60% of budget)
        # Phase 2: exploitation (remaining 40%)

        phase1_budget = int(budget * 0.6)
        phase2_budget = budget - phase1_budget

        def de_step(used_count, phase):
            """One DE step. Returns (new_x, new_f, used_count)."""
            if phase == 1:
                CR = 0.8
                F_mut = 0.6
                archive_prob = 0.4
            else:
                CR = 0.5
                F_mut = 0.35
                archive_prob = 0.75

            # Select base
            if len(archive_X) >= 2 and rng.random() < archive_prob:
                base_idx = rng.integers(len(archive_X))
                base_x = archive_X[base_idx]
            else:
                base_idx = rng.integers(pop_size)
                base_x = X[base_idx]

            # Select two parents from archive if possible
            if len(archive_X) >= 2:
                o1, o2 = rng.choice(len(archive_X), size=2, replace=False)
                x1 = archive_X[o1]
                x2 = archive_X[o2]
            else:
                o1, o2 = rng.choice(pop_size, size=2, replace=False)
                x1 = X[o1]
                x2 = X[o2]

            # Mutation
            mutant = base_x + F_mut * (x1 - x2)
            mutant = np.clip(mutant, lo, hi)

            # Crossover
            trial = base_x.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            f_trial = np.array(problem(trial))
            used_count += 1

            update_archive(trial, f_trial)

            # Population update: replace base if trial is non-dominated and better
            if base_idx < pop_size:
                base_f = F[base_idx]
                if _dominates(f_trial, base_f):
                    X[base_idx] = trial
                    F[base_idx] = f_trial
                elif not _dominates(base_f, f_trial):
                    # Non-dominated: accept with probability based on sum
                    if np.sum(f_trial) < np.sum(base_f) or rng.random() < 0.3:
                        X[base_idx] = trial
                        F[base_idx] = f_trial

            return used_count

        # Phase 1: exploration
        while used < phase1_budget:
            if time.time() - start_time > DEADLINE:
                break
            used = de_step(used, 1)

            # Periodic random injection for diversity
            if used % 15 == 0 and used < phase1_budget:
                if time.time() - start_time > DEADLINE:
                    break
                rand_x = lo + span * rng.random(dim)
                rand_f = np.array(problem(rand_x))
                used += 1
                update_archive(rand_x, rand_f)
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # Phase 2: exploitation - local search around best archive points
        if used < budget and archive_X:
            # Sort archive by sum, take top points
            sums = np.array([np.sum(af) for af in archive_F])
            top_n = min(20, len(archive_X))
            top_idx = np.argsort(sums)[:top_n]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Adaptive local search: try multiple scales
                sigmas = [0.12, 0.06, 0.03, 0.015]
                for sigma_scale in sigmas:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break

                    sigma = sigma_scale * span
                    n_trials = 4
                    for _ in range(n_trials):
                        if used >= budget:
                            break
                        if time.time() - start_time > DEADLINE:
                            break

                        perturbed = center + rng.normal(0, sigma, dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        f_pert = np.array(problem(perturbed))
                        used += 1

                        # Accept if dominates center
                        if _dominates(f_pert, center_f):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert
                        # Or if non-dominated and significantly better sum
                        elif not _dominates(center_f, f_pert):
                            if np.sum(f_pert) < np.sum(center_f) * 0.9:
                                update_archive(perturbed, f_pert)
                                center = perturbed
                                center_f = f_pert

        return X
# EVOLVE-BLOCK-END
