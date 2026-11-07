"""Multi-objective optimizer for ZDT4 with NSGA-II and deadline guard."""
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 330.0  # wall-clock limit
        safety = 10.0     # seconds before deadline to stop

        n = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        # Population size: use a good fraction of budget
        pop_size = max(50, min(200, budget // 2))
        # Ensure we can do at least a few generations
        if budget < pop_size * 2:
            pop_size = max(10, budget // 2)

        # --- Importance sampling for x_0 ---
        # ZDT4 Pareto front: f1 = x0^2, f2 = 1 + 10*(n-1) + (f1-1)^2*(1 - sqrt(|f1-1|/10))
        # Hypervolume contribution is higher where the front is flatter.
        # We sample x_0 with density proportional to the local hypervolume contribution.
        # Approximate: sample f1 from a distribution, then x0 = +/- sqrt(f1)
        # The Pareto front in f1-f2 space: f2(f1) for f1 in [0, 25]
        # HV contribution ~ 1 / |df2/df1| (roughly)
        # df2/df1 = 2*(f1-1)*(1 - sqrt(|f1-1|/10)) + (f1-1)^2 * (sign(f1-1)/(2*sqrt(10*|f1-1|)))
        # This is complex; use a simpler heuristic: sample f1 from a beta-like distribution
        # that emphasizes regions where the front has moderate curvature.

        def sample_x0(rng):
            """Sample x0 with density weighted by hypervolume contribution."""
            # f1 = x0^2, x0 in [-5, 5], f1 in [0, 25]
            # Sample f1 from a distribution that emphasizes the "interesting" part
            # The front is smooth for f1 > 1, has a cusp at f1=1
            # Use a mixture: 60% uniform on [0,25], 40% concentrated near [0.5, 5]
            if rng.random() < 0.6:
                f1 = rng.uniform(0, 25)
            else:
                f1 = rng.uniform(0.1, 8)
            f1 = np.clip(f1, 0, 25)
            x0 = np.sqrt(f1)
            if rng.random() < 0.5:
                x0 = -x0
            return x0

        # --- Initialize population ---
        X = np.empty((pop_size, n))
        for i in range(pop_size):
            x = np.empty(n)
            x[0] = sample_x0(rng := np.random)
            # For ZDT4, variables 1..n-1 should be near the optimum
            # The optimum for x_i (i>=1) is where 5*sin(10*x_i^2) = 0, i.e., x_i = 0, +/-pi/2, etc.
            # But for the Pareto front, we need the "global" optimum of the other objectives.
            # Actually for ZDT4, the Pareto front requires x_i to be at the global minimum
            # of the sin terms, which is x_i = 0 (since 5*sin(0)=0 and that's a min of |sin|).
            # Wait: ZDT4: g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} (x_i^2 - 10*cos(4*pi*x_i))
            # The minimum of g is when each x_i^2 - 10*cos(4*pi*x_i) is minimized.
            # At x_i=0: 0 - 10*1 = -10. At x_i=pi/2: pi^2/4 - 10*cos(2*pi) = 2.467 - 10 = -7.53
            # So x_i=0 gives the global minimum.
            # But the local minima are at x_i = 0, +/-pi/2, +/-pi, etc.
            # For the Pareto front, we need the GLOBAL minimum, so x_i should be near 0.
            for j in range(1, n):
                # Sample near 0 with some spread to escape local minima
                if rng.random() < 0.7:
                    x[j] = rng.uniform(-1, 1)
                else:
                    x[j] = rng.uniform(lo[j], hi[j])
            X[i] = x

        F = np.array([problem(X[i]) for i in range(pop_size)])
        used = pop_size

        # --- NSGA-II main loop ---
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            m = len(F)
            S = [set() for _ in range(m)]  # S[i] = indices dominated by i
            n_d = np.zeros(m, dtype=int)   # number of solutions that dominate i
            rank = np.zeros(m, dtype=int)
            fronts = []
            current_front = []
            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                if n_d[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            fronts.append(current_front)
            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                current_front = next_front
                if current_front:
                    fronts.append(current_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            m = len(front_indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(n_obj):
                f_vals = F[front_indices, k]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max == f_min:
                    continue
                order = np.argsort(f_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / (f_max - f_min)
            return cd

        def tournament_select(pop_X, pop_F, k=2):
            """Tournament selection based on rank and crowding distance."""
            m = len(pop_F)
            fronts = fast_non_dominated_sort(pop_F)
            rank = np.zeros(m, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Compute crowding distance per front
            cd = np.zeros(m)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(pop_F, front)
            # Tournament
            candidates = np.random.randint(0, m, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return pop_X[best].copy()

        def mutate(x, sigma):
            """Gaussian mutation with bounds clipping."""
            x_new = x + sigma * rng.standard_normal(n)
            return np.clip(x_new, lo, hi)

        def crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.8:
                # SBX
                x_c = np.empty(n)
                for j in range(n):
                    if x1[j] == x2[j]:
                        x_c[j] = x1[j]
                    else:
                        if rng.random() < 0.5:
                            x_c[j] = (x1[j] + x2[j]) / 2
                        else:
                            # Simple: average with some randomness
                            x_c[j] = rng.uniform(min(x1[j], x2[j]), max(x1[j], x2[j]))
                return x_c
            else:
                return x1.copy()

        # Main loop
        generation = 0
        while used < budget:
            if time.time() - t0 > deadline - safety:
                break

            # Selection and reproduction
            offspring_X = np.empty((pop_size, n))
            for i in range(pop_size):
                if time.time() - t0 > deadline - safety:
                    break
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                child = crossover(p1, p2)
                # Mutation: adapt sigma
                sigma = 0.1 * (1.0 - 0.5 * (used / budget))
                child = mutate(child, sigma)
                offspring_X[i] = child

            # Evaluate offspring
            offspring_F = np.array([problem(offspring_X[i]) for i in range(pop_size)])
            used += pop_size

            if used > budget:
                # Trim to budget
                offspring_X = offspring_X[:budget - (used - pop_size)]
                offspring_F = offspring_F[:budget - (used - pop_size)]
                pop_size = len(offspring_X)
                break

            # Combine and select
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            all_size = len(all_F)

            fronts = fast_non_dominated_sort(all_F)
            new_X = np.empty((0, n))
            new_F = np.empty((0, n_obj))
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X = np.vstack([new_X, all_X[front]]) if len(new_X) > 0 else all_X[front]
                    new_F = np.vstack([new_F, all_F[front]]) if len(new_F) > 0 else all_F[front]
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:remaining]]
                    if len(new_X) == 0:
                        new_X = all_X[selected]
                        new_F = all_F[selected]
                    else:
                        new_X = np.vstack([new_X, all_X[selected]])
                        new_F = np.vstack([new_F, all_F[selected]])
                    break

            X = new_X
            F = new_F
            generation += 1

        # --- Polish pass: local search on the Pareto front ---
        if time.time() - t0 < deadline - safety and used < budget:
            # Identify non-dominated points
            fronts = fast_non_dominated_sort(F)
            if fronts:
                pf_indices = list(fronts[0])
                # Try to improve each point with a few local steps
                for idx in pf_indices:
                    if time.time() - t0 > deadline - safety or used >= budget:
                        break
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    # Try small perturbations
                    for _ in range(5):
                        if used >= budget:
                            break
                        perturbed = best_x + 0.05 * rng.standard_normal(n)
                        perturbed = np.clip(perturbed, lo, hi)
                        pf = problem(perturbed)
                        used += 1
                        # Accept if it improves one objective without worsening the other
                        if (pf[0] < best_f[0] and pf[1] <= best_f[1] + 1e-6) or \
                           (pf[1] < best_f[1] and pf[0] <= best_f[0] + 1e-6):
                            best_x = perturbed
                            best_f = pf
                    X[idx] = best_x
                    F[idx] = best_f

        return X
