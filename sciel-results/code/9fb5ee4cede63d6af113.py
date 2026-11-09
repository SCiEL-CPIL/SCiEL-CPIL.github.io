"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with archive-guided search and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 325.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # --- Initialization ---
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
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_idx = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_idx])

        for sp in structured_extra:
            if len(init_points) < pop_size:
                init_points.append(sp)

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
            if not archive_F:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return
            A = np.array(archive_F)
            dominated = np.all(A <= f, axis=1) & np.any(A < f, axis=1)
            if np.any(dominated):
                return
            dominates = np.all(f <= A, axis=1) & np.any(f < A, axis=1)
            keep = ~dominates
            archive_X = [archive_X[i] for i in range(len(archive_X)) if keep[i]]
            archive_F = [archive_F[i] for i in range(len(archive_F)) if keep[i]]
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Fast non-dominated sorting ---
        def fast_nondominated_sort(F_mat):
            """Return list of fronts (lists of indices) and crowding distances."""
            n = len(F_mat)
            S = [[] for _ in range(n)]  # domination relation
            n_dom = [0] * n  # domination count
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            # Remove trailing empty fronts
            while fronts and not fronts[-1]:
                fronts.pop()

            # Crowding distances
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowd[idx] = np.inf
                    continue
                for m in range(n_obj):
                    sorted_idx = sorted(front, key=lambda k: F_mat[k][m])
                    crowd[sorted_idx[0]] = np.inf
                    crowd[sorted_idx[-1]] = np.inf
                    f_min = F_mat[sorted_idx[0]][m]
                    f_max = F_mat[sorted_idx[-1]][m]
                    if f_max - f_min < 1e-12:
                        continue
                    for k in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[k]
                        crowd[idx] += (F_mat[sorted_idx[k + 1]][m] - F_mat[sorted_idx[k - 1]][m]) / (f_max - f_min)

            return fronts, crowd

        # --- Main NSGA-II loop ---
        main_budget = budget - max(40, budget // 6)

        while used < main_budget:
            if time.time() - start_time > DEADLINE:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() - start_time > DEADLINE:
                    break

                # Tournament selection from population
                def tournament_select(pop_X, pop_F, k=2):
                    idxs = rng.choice(len(pop_X), size=k, replace=False)
                    best = idxs[0]
                    for idx in idxs[1:]:
                        if np.all(pop_F[idx] <= pop_F[best]) and np.any(pop_F[idx] < pop_F[best]):
                            best = idx
                    return best

                idx1 = tournament_select(X, F)
                idx2 = tournament_select(X, F)

                # SBX crossover
                if rng.random() < 0.9:
                    eta = 15.0
                    child = X[idx1].copy()
                    for j in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 1.0 / (eta + 1):
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1))
                            child[j] = 0.5 * ((1 + beta) * X[idx1][j] + (1 - beta) * X[idx2][j])
                else:
                    child = X[idx1].copy()

                # Polynomial mutation
                if rng.random() < 1.0 / dim:
                    j = rng.integers(dim)
                    u = rng.random()
                    eta_m = 20.0
                    if u <= 1.0 / (eta_m + 1):
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1))
                    child[j] += delta * (hi[j] - lo[j])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine population and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)

            # Non-dominated sorting
            fronts, crowd = fast_nondominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Sort by crowding distance and fill
                    remaining = pop_size - len(new_X)
                    if remaining > 0:
                        sorted_front = sorted(front, key=lambda k: crowd[k], reverse=True)
                        for idx in sorted_front[:remaining]:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

            # Update archive with new population
            for i in range(len(X)):
                update_archive(X[i], F[i])

            # Periodic diversity injection
            if used % 30 == 0 and used < main_budget:
                if time.time() - start_time > DEADLINE:
                    break
                if archive_X and rng.random() < 0.6:
                    base_arch = rng.integers(len(archive_X))
                    rand_x = archive_X[base_arch] + rng.normal(0, 0.25 * span, dim)
                else:
                    rand_x = lo + span * rng.random(dim)
                rand_x = np.clip(rand_x, lo, hi)
                rand_f = problem(rand_x)
                used += 1
                update_archive(rand_x, rand_f)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # --- Polish phase: local search on archive points ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

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

                # Multi-scale local search
                sigmas = [0.12, 0.06, 0.03, 0.015, 0.008]
                for sigma_scale in sigmas:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break

                    sigma = sigma_scale * span
                    for _ in range(4):
                        if used >= budget:
                            break
                        if time.time() - start_time > DEADLINE:
                            break
                        perturbed = center + rng.normal(0, sigma, dim)
                        perturbed = np.clip(perturbed, lo, hi)

                        f_pert = problem(perturbed)
                        used += 1

                        if np.all(f_pert <= center_f) and np.any(f_pert < center_f):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert
                        elif not (np.all(center_f <= f_pert) and np.any(center_f < f_pert)):
                            if np.sum(f_pert) < np.sum(center_f) * 0.97:
                                update_archive(perturbed, f_pert)
                                center = perturbed
                                center_f = f_pert

        return X
# EVOLVE-BLOCK-END
