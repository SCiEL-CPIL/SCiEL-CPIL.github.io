# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using a two-phase strategy:
    1) NSGA-II style global search to build an initial non-dominated set.
    2) A local search phase that refines points by projecting onto the
       estimated Pareto front tangent and doing small steps along it."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def sample():
            return lo + span * rng.random(self.dim)

        # Phase 1: NSGA-II-like population-based search
        pop_size = 40
        X = np.array([sample() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(F, indices):
            front = []
            for i in indices:
                is_dom = False
                for j in indices:
                    if j == i:
                        continue
                    if dominates(F[j], F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    front.append(i)
            return np.array(front)

        def crowding_distance(F, front_idx):
            if len(front_idx) <= 2:
                return np.ones(len(front_idx))
            cd = np.zeros(len(front_idx))
            for k in range(self.n_obj):
                vals = F[front_idx, k]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = 1.0
                    continue
                order = np.argsort(vals)
                cd[order[0]] += 1e10
                cd[order[-1]] += 1e10
                for i in range(1, len(order) - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (mx - mn)
            return cd

        while used < self.budget - 40:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = []
            remaining = list(range(len(F)))
            while remaining:
                front = non_dominated_front(F, remaining)
                fronts.append(front)
                remaining = [i for i in remaining if i not in set(front.tolist())]

            # Select parents: binary tournament on front rank + crowding
            parents = []
            for _ in range(2):
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                # find ranks
                ri = next(k for k, f in enumerate(fronts) if i in set(f.tolist()))
                rj = next(k for k, f in enumerate(fronts) if j in set(f.tolist()))
                if ri < rj:
                    parents.append(i)
                elif rj < ri:
                    parents.append(j)
                else:
                    # same front, use crowding
                    front = fronts[ri]
                    cd = crowding_distance(F, front)
                    ci = cd[front.tolist().index(i)]
                    cj = cd[front.tolist().index(j)]
                    parents.append(i if ci > cj else j)

            # Crossover + mutation
            p1, p2 = X[parents[0]], X[parents[1]]
            # Simulated binary crossover
            child = p1.copy()
            for d in range(self.dim):
                if rng.random() < 0.5:
                    # blend
                    t = rng.random()
                    child[d] = (1 - t) * p1[d] + t * p2[d]
            # Gaussian mutation
            child += 0.05 * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst in its front
            # Find the point with worst crowding in the last front
            last_front = fronts[-1]
            if len(last_front) > 0:
                cd = crowding_distance(F, last_front)
                worst_pos = last_front[np.argmin(cd)]
            else:
                worst_pos = int(np.argmax(np.sum(F, axis=1)))
            X[worst_pos] = child
            F[worst_pos] = fc

        # Phase 2: Local refinement along estimated Pareto front direction
        # Extract non-dominated points
        nd_front = non_dominated_front(F, list(range(len(F))))
        if len(nd_front) >= 2:
            # Estimate tangent direction in objective space
            F_nd = F[nd_front]
            # Principal direction
            F_centered = F_nd - F_nd.mean(axis=0)
            u, s, vt = np.linalg.svd(F_centered, full_matrices=False)
            tangent_obj = u[0]  # principal direction in objective space

            # For each ND point, try small perturbations in decision space
            # and keep improvements
            for i in nd_front:
                if time.time() > deadline or used >= self.budget:
                    break
                best_x = X[i].copy()
                best_f = F[i].copy()
                # Try several random perturbations
                for _ in range(8):
                    if used >= self.budget:
                        break
                    pert = 0.02 * span * rng.standard_normal(self.dim)
                    cand = np.clip(best_x + pert, lo, hi)
                    cf = problem(cand)
                    used += 1
                    # Check if cand is better in at least one objective and not worse in others
                    if np.all(cf <= best_f + 1e-10) and np.any(cf < best_f - 1e-10):
                        best_x = cand
                        best_f = cf
                X[i] = best_x
                F[i] = best_f

        return X
# EVOLVE-BLOCK-END
