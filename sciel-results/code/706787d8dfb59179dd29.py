import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        ref = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        pop_size = 60
        eta_c = 20.0
        eta_m = 20.0
        p_c = 0.95
        p_m = 1.0 / dim

        # Latin Hypercube Sampling for initialization
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])

        F = np.array([problem(x) for x in X])
        evals = pop_size

        def fast_non_dominated_sort(F):
            M = F.shape[0]
            S = [set() for _ in range(M)]
            n_dom = [0] * M
            fronts = []
            first_front = []

            for p in range(M):
                for q in range(p + 1, M):
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                        n_dom[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].add(p)
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    first_front.append(p)

            front = first_front
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, float('inf'))
            cd = np.zeros(m)
            k = F.shape[1]
            for j in range(k):
                order = np.argsort(F[indices, j])
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                rng = F[indices[order[-1]], j] - F[indices[order[0]], j]
                if rng > 1e-12:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F[indices[order[i+1]], j] - F[indices[order[i-1]], j]) / rng
            return cd

        def rank_and_select(F_comb, X_comb, n_select):
            fronts = fast_non_dominated_sort(F_comb)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    cd = crowding_distance(F_comb, front)
                    order = np.argsort(-cd)
                    need = n_select - len(selected)
                    selected.extend(front[order[:need]])
                    break
            return selected

        def sbx_crossover(x1, x2):
            if np.random.rand() < p_c:
                u = np.random.rand(dim)
                beta = np.ones(dim)
                mask1 = u < 0.5
                mask2 = ~mask1
                beta[mask1] = (2 * u[mask1]) ** (1.0 / (eta_c + 1))
                beta[mask2] = (1.0 / (2 * u[mask2])) ** (1.0 / (eta_c + 1))
                c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            return x1.copy(), x2.copy()

        def polynomial_mutation(x):
            for d in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u + 1) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u) + 1) ** (1.0 / (eta_m + 1))
                    x[d] = np.clip(x[d] + delta * (hi[d] - lo[d]), lo[d], hi[d])
            return x

        def hypervolume_contribution(F, idx, ref):
            point = F[idx]
            if np.any(point >= ref):
                return 0.0
            n = len(F)
            contrib = 1.0
            for d in range(len(point)):
                min_val = ref[d]
                for j in range(n):
                    if j != idx and F[j][d] < point[d]:
                        if F[j][d] > min_val:
                            min_val = F[j][d]
                gap = min_val - point[d]
                if gap > 0:
                    contrib *= gap
                else:
                    return 0.0
            return contrib

        def tournament_select_hv(F, size=2):
            indices = np.random.randint(0, len(F), size)
            best = indices[0]
            best_hv = hypervolume_contribution(F, best, ref)
            for i in indices[1:]:
                hv = hypervolume_contribution(F, i, ref)
                if hv > best_hv:
                    best = i
                    best_hv = hv
            return best

        # Main evolutionary loop
        while evals < budget:
            if time.time() > deadline:
                break

            X_off = np.zeros((pop_size, dim))
            F_off = np.zeros((pop_size, n_obj))

            for i in range(0, pop_size, 2):
                if time.time() > deadline:
                    break
                p1 = tournament_select_hv(F)
                p2 = tournament_select_hv(F)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                X_off[i] = c1
                X_off[i + 1] = c2
                F_off[i] = problem(c1)
                F_off[i + 1] = problem(c2)
                evals += 2

            if time.time() > deadline:
                break

            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])

            selected = rank_and_select(F_comb, X_comb, pop_size)
            X = X_comb[selected]
            F = F_comb[selected]

        # Local refinement of the best non-dominated points
        if evals < budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            if fronts:
                best_front = fronts[0]
                front_hvs = [(hypervolume_contribution(F, idx, ref), idx) for idx in best_front]
                front_hvs.sort(reverse=True)
                n_refine = min(10, len(front_hvs), max(1, (budget - evals) // 10))

                for _, idx in front_hvs[:n_refine]:
                    if evals >= budget or time.time() > deadline:
                        break
                    x0 = X[idx].copy()
                    f0 = F[idx].copy()
                    base_step = 0.02 * (hi - lo)

                    for iter_count in range(5):
                        if evals >= budget or time.time() > deadline:
                            break
                        step_size = base_step * (0.5 ** iter_count)
                        improved = False
                        for d in range(dim):
                            if evals >= budget or time.time() > deadline:
                                break
                            for direction in [1, -1]:
                                if evals >= budget or time.time() > deadline:
                                    break
                                x_trial = x0.copy()
                                x_trial[d] += direction * step_size[d]
                                x_trial = np.clip(x_trial, lo, hi)
                                f_trial = problem(x_trial)
                                evals += 1
                                if np.all(f_trial <= f0) and np.any(f_trial < f0):
                                    x0 = x_trial
                                    f0 = f_trial
                                    improved = True
                                    break
                                elif not (np.all(f0 <= f_trial) and np.any(f0 < f_trial)):
                                    if np.all(f_trial < ref) and np.all(f0 < ref):
                                        norm_f0 = f0 / ref
                                        norm_ft = f_trial / ref
                                        if np.sum(norm_ft) < np.sum(norm_f0):
                                            x0 = x_trial
                                            f0 = f_trial
                                            improved = True
                                            break
                        if not improved:
                            break
                    X[idx] = x0
                    F[idx] = f0

        return F
