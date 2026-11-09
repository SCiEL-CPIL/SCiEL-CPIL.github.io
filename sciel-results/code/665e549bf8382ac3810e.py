<<<<<<< SEARCH
"""Multi-objective optimizer for RE21: HV-driven evolutionary search + adaptive mutation + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven multi-objective search with adaptive mutation and L-BFGS-B local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        def dominated_mask(Fv):
            """mask[i] = True if Fv[i] is dominated by some other row."""
            n = len(Fv)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                d = (Fv <= Fv[i]) & (Fv < Fv[i])
                if d.sum() > 1:  # at least one other row dominates i
                    dom[i] = True
            return dom

        def compute_hv_contrib(Fv, nd_idx, ref):
            """Compute HV contribution for non-dominated points."""
            nd = Fv[nd_idx]
            n_nd = len(nd)
            hv_contrib = np.zeros(n_nd)
            if n_nd == 0:
                return hv_contrib
            order = np.argsort(nd[:, 0])
            nd_sorted = nd[order]
            prev_f1 = ref[1]
            for k in range(n_nd - 1, -1, -1):
                f0, f1 = nd_sorted[k]
                if f0 < ref[0] and f1 < ref[1]:
                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                    hv_contrib[order[k]] = max(contrib, 0.0)
                else:
                    hv_contrib[order[k]] = 0.0
                prev_f1 = f1
            return hv_contrib

        # 1) Random initialization with boundary focus
        n_init = 20
        rng = np.random.default_rng(12345)
        for i in range(n_init):
            if i < n_init // 2:
                x = lo + span * rng.random(dim)
            else:
                # Boundary-focused: force some dims to be near bounds
                u = rng.random(dim)
                mask = rng.random(dim) < 0.3
                if np.any(mask):
                    u[mask] = rng.random(np.sum(mask))
                x = lo + span * u
            X.append(x)
            F.append(eval_pt(x))
        if used >= self.budget:
            return np.array(X)

        # 2) Evolutionary loop with adaptive sigma
        sigma = 0.15
        min_sigma = 0.01
        max_sigma = 0.3
        stagnation = 0
        best_hv = 0.0
        ref = np.array([3000.0, 0.06])

        while used < self.budget:
            if time.time() > deadline - 20.0:  # reserve time for polish
                break

            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Compute HV contribution (2D) for each non-dominated point
            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)

            total_hv = hv_contrib.sum()
            if total_hv > best_hv:
                best_hv = total_hv
                stagnation = 0
            else:
                stagnation += 1

            # Adaptive sigma: grow if stagnant, shrink if improving
            if stagnation > 30:
                sigma = min(max_sigma, sigma * 1.5)
                stagnation = 0
            elif total_hv > best_hv * 0.95:
                sigma = max(min_sigma, sigma * 0.97)

            # Selection: HV-contribution driven
            if total_hv <= 0:
                p = rng.integers(len(nd_idx))
            else:
                probs = hv_contrib / total_hv
                # Add small epsilon to avoid zero-probability issues
                probs = probs + 1e-10
                probs /= probs.sum()
                p = rng.choice(len(nd_idx), p=probs)

            parent = Xv[nd_idx[p]].copy()

            # Mutation
            step = sigma * span * rng.standard_normal(dim)
            child = np.clip(parent + step, lo, hi)
            fc = eval_pt(child)

            # Replace the point with the lowest HV contribution
            worst = int(np.argmin(hv_contrib))
            X[nd_idx[worst]] = child
            F[nd_idx[worst]] = fc

        # 3) Local polish: L-BFGS-B on top HV-contributing points with weighted-sum objectives
        if time.time() < deadline - 5.0 and used < self.budget:
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)
            n_nd = len(nd_idx)

            # Use top 5 HV contributors as starting points for L-BFGS-B
            top_k = min(5, n_nd, max(1, (self.budget - used) // 2))
            top_indices = np.argsort(hv_contrib)[::-1][:top_k]

            # Weight vectors for weighted-sum (tuned for diversity)
            n_w = min(8, max(1, (self.budget - used) // top_k))
            weights = np.array([[1.0 - i / max(n_w - 1, 1), i / max(n_w - 1, 1)]
                                for i in range(n_w)])

            for w in weights:
                if used >= self.budget or time.time() > deadline - 2.0:
                    break

                def obj_func(x, w=w):
                    f = np.asarray(problem(x), dtype=float)
                    return float(np.dot(w, f))

                # For each weight, find best starting point among top-k
                Fv2 = np.array(F)
                scores = Fv2[nd_idx] @ w
                best_local = nd_idx[int(np.argmin(scores))]
                x0 = Xv[best_local].copy()

                try:
                    res = minimize(obj_func, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                                            'maxls': 50})
                    if res.success and used < self.budget:
                        xf = np.clip(res.x, lo, hi)
                        ff = eval_pt(xf)
                        # Replace worst HV point
                        Fv3 = np.array(F)
                        dom3 = dominated_mask(Fv3)
                        nd3 = np.where(~dom3)[0]
                        if len(nd3) > 0:
                            hv3 = compute_hv_contrib(Fv3, nd3, ref)
                            worst3 = int(np.argmin(hv3))
                            X[nd3[worst3]] = xf
                            F[nd3[worst3]] = ff
                except Exception:
                    pass

        return np.array(X)
# EVOLVE-BLOCK-END
=======
"""Multi-objective optimizer for RE21: NSGA-II with crowding distance + adaptive mutation + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II style multi-objective search with crowding distance and L-BFGS-B local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        def non_dominated_sort(Fv):
            """Return (fronts, rank) where fronts is list of index lists, rank[i] = front number."""
            n = len(Fv)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(Fv[i] <= Fv[j]) and np.any(Fv[i] < Fv[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(Fv[j] <= Fv[i]) and np.any(Fv[j] < Fv[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            front0 = [i for i in range(n) if n_dom[i] == 0]
            if not front0:
                return [list(range(n))], rank
            fronts.append(front0)
            for i in front0:
                rank[i] = 0

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts, rank

        def crowding_distance(Fv, front_idx):
            """Compute crowding distance for points in a front."""
            n = len(front_idx)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            Ffront = Fv[front_idx]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(Ffront[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = Ffront[sorted_idx[-1], m] - Ffront[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (Ffront[sorted_idx[i + 1], m] - Ffront[sorted_idx[i - 1], m]) / f_range

            return cd

        def compute_hv_contrib(Fv, nd_idx, ref):
            """Compute HV contribution for non-dominated points."""
            nd = Fv[nd_idx]
            n_nd = len(nd)
            hv_contrib = np.zeros(n_nd)
            if n_nd == 0:
                return hv_contrib
            order = np.argsort(nd[:, 0])
            nd_sorted = nd[order]
            prev_f1 = ref[1]
            for k in range(n_nd - 1, -1, -1):
                f0, f1 = nd_sorted[k]
                if f0 < ref[0] and f1 < ref[1]:
                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                    hv_contrib[order[k]] = max(contrib, 0.0)
                else:
                    hv_contrib[order[k]] = 0.0
                prev_f1 = f1
            return hv_contrib

        # 1) Random initialization with diverse sampling
        n_init = 24
        rng = np.random.default_rng(12345)
        for i in range(n_init):
            # Mix of uniform and boundary-focused sampling
            if i < n_init // 3:
                x = lo + span * rng.random(dim)
            elif i < 2 * n_init // 3:
                # Boundary focus: some dims near bounds
                u = rng.random(dim)
                mask = rng.random(dim) < 0.4
                if np.any(mask):
                    u[mask] = rng.random(np.sum(mask)) * 0.1
                    u[mask] = np.where(rng.random(np.sum(mask)) < 0.5, 0.0, 1.0) * u[mask]
                x = lo + span * u
            else:
                # Latin hypercube-like: stratified sampling
                u = (np.arange(dim) + rng.random(dim)) / dim
                x = lo + span * u
            X.append(x)
            F.append(eval_pt(x))

        if used >= self.budget:
            return np.array(X)

        # 2) NSGA-II evolutionary loop
        sigma = 0.12
        min_sigma = 0.005
        max_sigma = 0.25
        stagnation = 0
        best_hv = 0.0
        ref = np.array([3000.0, 0.06])
        pop_size = min(30, self.budget - used)

        while used < self.budget:
            if time.time() > deadline - 30.0:  # reserve time for polish
                break

            Fv = np.array(F)
            Xv = np.array(X)
            n_pop = len(Fv)

            # Non-dominated sorting
            fronts, rank = non_dominated_sort(Fv)

            # Compute crowding distance for each front
            cd_all = np.zeros(n_pop)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(Fv, front)
                    for idx, cd_val in zip(front, cd):
                        cd_all[idx] = cd_val

            # Compute HV contribution for selection
            nd_idx = np.where(rank == 0)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(n_pop)
            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)

            total_hv = hv_contrib.sum()
            if total_hv > best_hv:
                best_hv = total_hv
                stagnation = 0
            else:
                stagnation += 1

            # Adaptive sigma
            if stagnation > 25:
                sigma = min(max_sigma, sigma * 1.4)
                stagnation = 0
            elif total_hv > best_hv * 0.98:
                sigma = max(min_sigma, sigma * 0.96)

            # Binary tournament selection using rank + crowding distance
            def tournament_select():
                i = rng.integers(n_pop)
                j = rng.integers(n_pop)
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd_all[i] > cd_all[j] else j

            # Generate offspring
            n_offspring = max(1, min(5, self.budget - used))
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                p1 = tournament_select()
                p2 = tournament_select()

                # SBX-like crossover (simple blend)
                if rng.random() < 0.8:
                    alpha = rng.random()
                    child = alpha * Xv[p1] + (1 - alpha) * Xv[p2]
                else:
                    child = Xv[p1].copy()

                # Gaussian mutation
                step = sigma * span * rng.standard_normal(dim)
                child = np.clip(child + step, lo, hi)

                offspring_X.append(child)
                offspring_F.append(eval_pt(child))

                if used >= self.budget:
                    break

            if not offspring_X:
                break

            # Combine population and offspring
            combined_X = np.vstack([Xv, np.array(offspring_X)])
            combined_F = np.vstack([Fv, np.array(offspring_F)])
            n_combined = len(combined_F)

            # Non-dominated sorting on combined
            fronts_c, rank_c = non_dominated_sort(combined_F)

            # Select best pop_size individuals
            selected = []
            for front in fronts_c:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(selected)
                    cd_front = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(cd_front)[::-1]
                    selected.extend([front[i] for i in sorted_front[:remaining]])
                    break

            X = [combined_X[i] for i in selected]
            F = [combined_F[i] for i in selected]

        # 3) Local polish: L-BFGS-B on top points with diverse weight vectors
        if time.time() < deadline - 10.0 and used < self.budget:
            Fv = np.array(F)
            Xv = np.array(X)
            fronts, rank = non_dominated_sort(Fv)

            # Use non-dominated points as starting points
            nd_idx = np.where(rank == 0)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Compute HV contribution for prioritization
            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)
            n_nd = len(nd_idx)

            # Weight vectors for weighted-sum (evenly spaced)
            n_w = min(12, max(1, (self.budget - used) // 3))
            weights = []
            for i in range(n_w):
                w1 = i / max(n_w - 1, 1)
                w2 = 1.0 - w1
                weights.append(np.array([w1, w2]))

            for w in weights:
                if used >= self.budget or time.time() > deadline - 5.0:
                    break

                def obj_func(x, w=w):
                    f = np.asarray(problem(x), dtype=float)
                    return float(np.dot(w, f))

                # Find best starting point for this weight
                Fv2 = np.array(F)
                scores = Fv2[nd_idx] @ w
                best_local = nd_idx[int(np.argmin(scores))]
                x0 = Xv[best_local].copy()

                try:
                    res = minimize(obj_func, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                            'maxls': 50})
                    if res.success and used < self.budget:
                        xf = np.clip(res.x, lo, hi)
                        ff = eval_pt(xf)

                        # Check if this improves HV
                        Fv3 = np.array(F)
                        fronts3, rank3 = non_dominated_sort(Fv3)
                        nd3 = np.where(rank3 == 0)[0]
                        if len(nd3) > 0:
                            hv3 = compute_hv_contrib(Fv3, nd3, ref)
                            # Replace worst HV point
                            worst3 = int(np.argmin(hv3))
                            X[nd3[worst3]] = xf
                            F[nd3[worst3]] = ff
                except Exception:
                    pass

        return np.array(X)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
