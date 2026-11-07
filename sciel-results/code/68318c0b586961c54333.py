import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1.1, 1.3, 1.1])

        eval_count = 0
        X_all = []
        F_all = []

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            try:
                f = np.asarray(problem(x), dtype=float).reshape(-1)[:n_obj]
            except Exception:
                return None
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # ---------- Pareto / hypervolume helpers ----------
        def non_dominated_indices(F):
            F = np.asarray(F)
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        def hypervolume_3d(F, ref_pt):
            """Exact 3D hypervolume of non-dominated set F w.r.t. ref_pt."""
            F = np.asarray(F, dtype=float)
            if len(F) == 0:
                return 0.0
            keep = np.all(F < ref_pt, axis=1)
            F = F[keep]
            if len(F) == 0:
                return 0.0
            idx = non_dominated_indices(F)
            F = F[idx]
            n = len(F)
            if n == 0:
                return 0.0

            order = np.argsort(F[:, 0])
            P = F[order]
            total = 0.0
            for i in range(n):
                p = P[i]
                next0 = P[i + 1][0] if i + 1 < n else ref_pt[0]
                w0 = next0 - p[0]
                if w0 <= 0:
                    continue
                p1, p2 = p[1], p[2]
                r1, r2 = ref_pt[1], ref_pt[2]
                area = (r1 - p1) * (r2 - p2)
                below = [(P[j][1], P[j][2]) for j in range(i) if P[j][1] < p1 and P[j][2] < p2]
                if below:
                    below_arr = np.array(below)
                    below_arr = below_arr[(below_arr[:, 0] < p1) & (below_arr[:, 1] < p2)]
                    if len(below_arr) > 0:
                        n2 = len(below_arr)
                        nd2 = np.ones(n2, dtype=bool)
                        for a in range(n2):
                            if not nd2[a]:
                                continue
                            for b in range(n2):
                                if a == b or not nd2[b]:
                                    continue
                                if (below_arr[b][0] <= below_arr[a][0] and 
                                    below_arr[b][1] <= below_arr[a][1] and 
                                    (below_arr[b][0] < below_arr[a][0] or below_arr[b][1] < below_arr[a][1])):
                                    nd2[a] = False
                                    break
                        below_arr = below_arr[nd2]
                        below_arr = below_arr[np.argsort(below_arr[:, 0])]
                        hv2 = 0.0
                        for k in range(len(below_arr)):
                            bk = below_arr[k]
                            next1 = below_arr[k + 1][0] if k + 1 < len(below_arr) else p1
                            w1 = next1 - bk[0]
                            if w1 <= 0:
                                continue
                            h2 = p2 - bk[1]
                            for m in range(k):
                                bm = below_arr[m]
                                if bm[1] < bk[1]:
                                    h2 = min(h2, p2 - bm[1])
                            if h2 > 0:
                                hv2 += w1 * h2
                        area -= hv2
                if area > 0:
                    total += w0 * area
            return total

        def hv_contribution(f_new, nd_F):
            """Compute how much hypervolume f_new adds to the non-dominated set."""
            if len(nd_F) == 0:
                return hypervolume_3d(np.array([f_new]), ref)
            cand = np.vstack([nd_F, f_new])
            return hypervolume_3d(cand, ref) - hypervolume_3d(nd_F, ref)

        # ---------- Phase 1: LHS initialization ----------
        n_initial = min(30, budget)
        if n_initial > 0:
            samples = np.zeros((n_initial, dim))
            for d in range(dim):
                perm = np.random.permutation(n_initial)
                samples[:, d] = (perm + np.random.random(n_initial)) / n_initial
            samples = lo + (hi - lo) * samples
            for x in samples:
                if not time_ok():
                    break
                f = evaluate(x)
                if f is None:
                    break

        if len(X_all) == 0:
            x0 = lo + 0.5 * (hi - lo)
            f0 = evaluate(x0)
            if f0 is not None:
                return x0
            return x0

        F_current = np.array(F_all)
        nd_idx = non_dominated_indices(F_current)
        nd_F = F_current[nd_idx].copy() if len(nd_idx) > 0 else F_current.copy()
        base_hv = hypervolume_3d(nd_F, ref)

        # ---------- Phase 2: Hypervolume-guided NSGA-II ----------
        pop_size = 40

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, front):
            if len(front) <= 2:
                return {i: float('inf') for i in front}
            dist = {i: 0.0 for i in front}
            for m in range(n_obj):
                vals = sorted([(F[i][m], i) for i in front])
                dist[vals[0][1]] = float('inf')
                dist[vals[-1][1]] = float('inf')
                fmin = vals[0][0]
                fmax = vals[-1][0]
                if fmax - fmin < 1e-12:
                    continue
                for k in range(1, len(vals) - 1):
                    dist[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / (fmax - fmin)
            return dist

        # Build initial population from all evaluated points
        fronts, rank = fast_non_dominated_sort(F_current)
        pop_indices = []
        for front in fronts:
            if len(pop_indices) + len(front) <= pop_size:
                pop_indices.extend(front)
            else:
                cd = crowding_distance(F_current, front)
                sorted_front = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                remaining = pop_size - len(pop_indices)
                pop_indices.extend(sorted_front[:remaining])
                break
        while len(pop_indices) < pop_size:
            pop_indices.append(np.random.randint(len(F_all)))

        X = np.array([X_all[i] for i in pop_indices])
        F = np.array([F_all[i] for i in pop_indices])

        # Evolutionary loop with hypervolume-guided selection
        while time_ok():
            if eval_count >= budget - 1:
                break

            fronts, rank = fast_non_dominated_sort(F)
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(F, front))

            def tournament():
                i = np.random.randint(len(F))
                j = np.random.randint(len(F))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd.get(i, 0) > cd.get(j, 0) else j

            n_children = min(pop_size, budget - eval_count)
            children_X = []
            children_F = []

            for _ in range(n_children):
                if not time_ok():
                    break

                p1 = tournament()
                p2 = tournament()

                # SBX crossover
                if np.random.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if np.random.random() < 0.5:
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 17.0)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 17.0)
                            if X[p1][d] <= X[p2][d]:
                                child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p2][d] + (1 - beta) * X[p1][d])
                        else:
                            child[d] = X[p1][d]
                else:
                    child = X[p1].copy()

                # Mutation
                for d in range(dim):
                    if np.random.random() < 1.0 / dim:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (2 + 20))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (2 + 20))
                        if child[d] <= lo[d]:
                            child[d] = child[d] + (hi[d] - child[d]) * beta
                        elif child[d] >= hi[d]:
                            child[d] = child[d] - (child[d] - lo[d]) * beta
                        else:
                            child[d] = child[d] + (hi[d] - lo[d]) * (beta - 1) * np.random.choice([-1, 1])
                child = np.clip(child, lo, hi)

                f_new = evaluate(child)
                if f_new is None:
                    break
                children_X.append(child)
                children_F.append(f_new)

            if not children_X:
                break

            # Combine parent and child populations
            combined_X = np.vstack([X, np.array(children_X)])
            combined_F = np.vstack([F, np.array(children_F)])

            # Selection: keep best pop_size by non-dominated sort + crowding
            fronts_c, rank_c = fast_non_dominated_sort(combined_F)
            cd_c = {}
            for front in fronts_c:
                cd_c.update(crowding_distance(combined_F, front))

            new_pop_indices = []
            for front in fronts_c:
                if len(new_pop_indices) + len(front) <= pop_size:
                    new_pop_indices.extend(front)
                else:
                    sorted_front = sorted(front, key=lambda i: cd_c.get(i, 0), reverse=True)
                    remaining = pop_size - len(new_pop_indices)
                    new_pop_indices.extend(sorted_front[:remaining])
                    break

            X = combined_X[new_pop_indices]
            F = combined_F[new_pop_indices]

            # Update global non-dominated set
            F_current = np.array(F_all)
            nd_idx = non_dominated_indices(F_current)
            nd_F = F_current[nd_idx].copy() if len(nd_idx) > 0 else F_current.copy()
            new_hv = hypervolume_3d(nd_F, ref)

            # If stagnation detected, inject gap-filling points
            if new_hv <= base_hv + 1e-10 and time_ok():
                base_hv = new_hv
                # Find the largest gap in objective space
                if len(nd_F) >= 2:
                    # Compute bounding box of non-dominated set
                    obj_min = nd_F.min(axis=0)
                    obj_max = nd_F.max(axis=0)
                    # Try several random gap centers
                    for _ in range(5):
                        if not time_ok():
                            break
                        # Random point in the bounding box, perturbed
                        x_gap = lo + np.random.random(dim) * (hi - lo)
                        f_gap = evaluate(x_gap)
                        if f_gap is None:
                            break
                        cand = np.vstack([nd_F, f_gap])
                        hv_cand = hypervolume_3d(cand, ref)
                        if hv_cand > base_hv + 1e-10:
                            nd_idx2 = non_dominated_indices(cand)
                            nd_F = cand[nd_idx2]
                            base_hv = hv_cand
                            # Local refinement around the new point
                            for step in range(4):
                                if not time_ok():
                                    break
                                x_ref = x_gap + 0.1 * np.random.standard_normal(dim)
                                x_ref = np.clip(x_ref, lo, hi)
                                f_ref = evaluate(x_ref)
                                if f_ref is None:
                                    break
                                cand2 = np.vstack([nd_F, f_ref])
                                hv2 = hypervolume_3d(cand2, ref)
                                if hv2 > base_hv + 1e-10:
                                    nd_idx3 = non_dominated_indices(cand2)
                                    nd_F = cand2[nd_idx3]
                                    base_hv = hv2
                                    x_gap = x_ref

        # Return the best point
        if len(X_all) > 0:
            F_final = np.array(F_all)
            final_nd = non_dominated_indices(F_final)
            if len(final_nd) > 0:
                return X_all[final_nd[0]]
            return X_all[0]
        return lo + 0.5 * (hi - lo)
