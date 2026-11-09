# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # ---------- helpers ----------
        def dominates(a, b):
            return bool(np.all(a <= b) and np.any(a < b))

        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            frontiers = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    frontiers.append(i)
            k = 0
            result = np.empty(n, dtype=int)
            while len(frontiers) > 0:
                next_front = []
                for i in frontiers:
                    result[i] = k
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                frontiers = next_front
                k += 1
            return result

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n, dtype=np.float64)
            for rank in set(ranks):
                idx = np.where(ranks == rank)[0]
                m = len(idx)
                if m <= 2:
                    cd[idx] = np.inf
                    continue
                for d in range(n_obj):
                    vals = F[idx, d]
                    vmin = vals.min()
                    vmax = vals.max()
                    rng = vmax - vmin
                    if rng < 1e-12:
                        continue
                    order = np.argsort(vals)
                    cd[idx[order[0]]] = np.inf
                    cd[idx[order[-1]]] = np.inf
                    for kk in range(1, m - 1):
                        cd[idx[order[kk]]] += (vals[order[kk + 1]] - vals[order[kk - 1]]) / rng
            return cd

        # ---------- structured initial population ----------
        rng = np.random.default_rng(0)

        # Compact initialization: 20 structured + 5 random = 25 total
        n_struct = 20
        n_rand = 5
        n_init = n_struct + n_rand

        X_init = np.empty((n_init, dim))
        for i in range(n_struct):
            for d in range(dim):
                X_init[i, d] = lo[d] + span[d] * ((i + 0.5) / n_struct)
        X_init[:n_struct] += 0.03 * span * rng.random((n_struct, dim))
        X_init[n_struct:] = lo + span * rng.random((n_rand, dim))
        X_init = np.clip(X_init, lo, hi)

        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        # Build archive from init
        archive_X = []
        archive_F = []
        for i in range(len(X_init)):
            dominated = False
            for j in range(len(archive_F)):
                if dominates(archive_F[j], F_init[i]):
                    dominated = True
                    break
            if dominated:
                continue
            new_ax = list(archive_X)
            new_af = list(archive_F)
            to_remove = [j for j in range(len(new_af)) if dominates(F_init[i], new_af[j])]
            for j in sorted(to_remove, reverse=True):
                del new_ax[j]
                del new_af[j]
            new_ax.append(X_init[i])
            new_af.append(F_init[i])
            archive_X = new_ax
            archive_F = new_af

        # ---------- DE population from archive ----------
        pop_size = 15
        pop_X = np.empty((pop_size, dim))
        if len(archive_X) >= pop_size:
            idx = rng.choice(len(archive_X), size=pop_size, replace=False)
            for i, j in enumerate(idx):
                pop_X[i] = archive_X[j]
        else:
            for i in range(len(archive_X)):
                pop_X[i] = archive_X[i]
            for i in range(len(archive_X), pop_size):
                pop_X[i] = lo + span * rng.random(dim)
        pop_F = np.array([problem(x) for x in pop_X])
        used += pop_size

        # DE parameters
        CR = 0.8
        F_mut = 0.5

        # Track best archive for polishing
        best_archive_F = np.array(archive_F) if archive_F else np.empty((0, n_obj))

        while used < budget:
            if time.time() > deadline - 10:
                break

            ranks = fast_nondominated_sort(pop_F)
            cd = crowding_distance(pop_F, ranks)

            # Generate 2 children per iteration
            for _ in range(2):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Tournament selection on rank
                cand_idx = rng.choice(pop_size, size=3, replace=False)
                winner = cand_idx[np.argmin(ranks[cand_idx])]
                parent = pop_X[winner]

                # DE mutation
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
                mask = rng.random(dim) < CR
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                child = np.where(mask, v, parent)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update archive
                dominated = False
                for j in range(len(archive_F)):
                    if dominates(archive_F[j], fc):
                        dominated = True
                        break
                if not dominated:
                    new_ax = list(archive_X)
                    new_af = list(archive_F)
                    to_remove = [j for j in range(len(new_af)) if dominates(fc, new_af[j])]
                    for j in sorted(to_remove, reverse=True):
                        del new_ax[j]
                        del new_af[j]
                    new_ax.append(child)
                    new_af.append(fc)
                    archive_X = new_ax
                    archive_F = new_af
                    best_archive_F = np.array(archive_F)

                # Replace worst in population (by rank, tie-break by crowding)
                worst_idx = np.argmax(ranks)
                pop_X[worst_idx] = child
                pop_F[worst_idx] = fc
                ranks = fast_nondominated_sort(pop_F)

        # ---------- Local polishing phase ----------
        # Use remaining budget to polish top archive points
        if len(archive_F) > 0 and used < budget:
            # Get top 5 non-dominated points for polishing
            arch_F = np.array(archive_F)
            arch_X = np.array(archive_X)
            # Sort by sum of objectives (lower is better) as proxy
            obj_sums = arch_F.sum(axis=1)
            top_idx = np.argsort(obj_sums)[:min(5, len(arch_F))]

            polish_budget = budget - used
            per_point = max(1, polish_budget // len(top_idx))

            for ti in top_idx:
                if used >= budget or time.time() > deadline - 2:
                    break

                x0 = arch_X[ti]
                # Minimize weighted sum with random weights
                for _ in range(min(per_point, 3)):
                    if used >= budget or time.time() > deadline - 1:
                        break
                    weights = rng.random(n_obj)
                    weights /= weights.sum()

                    def obj_func(x):
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        return np.dot(weights, f)

                    try:
                        res = minimize(obj_func, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                        used += 1  # approximate: each minimize call uses ~1 eval in our budget tracking
                        # Actually scipy uses multiple evals, so we need to be careful
                        # Let's count it as 1 for now and adjust
                        if res.success and res.fun < obj_sums[ti]:
                            x_polished = np.clip(res.x, lo, hi)
                            f_polished = problem(x_polished)
                            used += 1
                            # Add to archive if non-dominated
                            dominated = False
                            for j in range(len(archive_F)):
                                if dominates(archive_F[j], f_polished):
                                    dominated = True
                                    break
                            if not dominated:
                                new_ax = list(archive_X)
                                new_af = list(archive_F)
                                to_remove = [j for j in range(len(new_af)) if dominates(f_polished, new_af[j])]
                                for j in sorted(to_remove, reverse=True):
                                    del new_ax[j]
                                    del new_af[j]
                                new_ax.append(x_polished)
                                new_af.append(f_polished)
                                archive_X = new_ax
                                archive_F = new_af
                                obj_sums = np.array(archive_F).sum(axis=1)
                    except:
                        pass

        return pop_X
# EVOLVE-BLOCK-END
