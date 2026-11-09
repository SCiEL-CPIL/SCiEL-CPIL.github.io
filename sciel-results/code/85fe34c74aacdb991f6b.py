"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using efficient NSGA-II with adaptive search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Efficient non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=np.int32)

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1

        front = [i for i in range(n) if nd[i] == 0]
        fronts = []
        if not front:
            front = [0]
        fronts.append(front)

        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf

            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue

            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population: mix of systematic points and random
        pop_size = min(50, max(25, self.budget // 4))

        # Create a diverse initial set: vertices, midpoints, random
        init_points = []
        # Vertices of the hypercube
        for i in range(2 ** self.dim):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                if (i >> d) & 1:
                    x[d] = 1.0
            init_points.append(x)
        # Center
        init_points.append(0.5 * np.ones(self.dim))
        # Additional structured points: face centers and edge midpoints
        for d in range(self.dim):
            x_low = np.full(self.dim, 0.5)
            x_low[d] = 0.0
            x_high = np.full(self.dim, 0.5)
            x_high[d] = 1.0
            init_points.append(x_low)
            init_points.append(x_high)
        # Random points
        while len(init_points) < pop_size:
            init_points.append(np.random.random(self.dim))

        X = np.array(init_points[:pop_size])
        X = np.clip(X, 0, 1)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best front
        all_X = X.copy()
        all_F = F.copy()

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting on current population
            fronts = self._fast_non_dominated_sort(F)

            # Build offspring
            offspring = []
            # Compute objective ranges for adaptive mutation
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_range = np.where(F_max > F_min, F_max - F_min, 1.0)

            while len(offspring) < pop_size:
                # Tournament selection
                p1 = self._tournament_select(F, fronts)
                p2 = self._tournament_select(F, fronts)

                # BLX-Alpha crossover with adaptive alpha
                if np.random.random() < 0.9:
                    progress = used / self.budget
                    alpha = 0.3 + 0.3 * (1.0 - progress)
                    c1 = np.zeros(self.dim)
                    c2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        x1 = X[p1, d]
                        x2 = X[p2, d]
                        low = min(x1, x2) - alpha * abs(x2 - x1)
                        high = max(x1, x2) + alpha * abs(x2 - x1)
                        low = max(low, 0.0)
                        high = min(high, 1.0)
                        c1[d] = low + (high - low) * np.random.random()
                        c2[d] = low + (high - low) * np.random.random()
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()

                # Adaptive mutation: larger when population is diverse in objectives
                for c in [c1, c2]:
                    if np.random.random() < 0.9:
                        # Mutation strength adapts: start large, shrink as we converge
                        progress = used / self.budget
                        strength = 0.25 * (1.0 - 0.6 * progress)
                        c += strength * np.random.standard_normal(self.dim)
                        c = np.clip(c, 0, 1)

                offspring.append(c1)
                offspring.append(c2)

            offspring = np.array(offspring[:pop_size])
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size

            # Combine and select
            X = np.vstack([X, offspring])
            F = np.vstack([F, F_off])

            fronts = self._fast_non_dominated_sort(F)
            next_gen_indices = []
            for front in fronts:
                if len(next_gen_indices) + len(front) <= pop_size:
                    next_gen_indices.extend(front)
                else:
                    need = pop_size - len(next_gen_indices)
                    dist = self._crowding_distance(F, front)
                    order = np.argsort(dist)[::-1]
                    next_gen_indices.extend([front[i] for i in order[:need]])
                    break

            X = X[next_gen_indices]
            F = F[next_gen_indices]

            # Keep track of all evaluated points for final hypervolume
            all_X = np.vstack([all_X, offspring])
            all_F = np.vstack([all_F, F_off])

        # Final local polish: refine the non-dominated front
        if used < self.budget and time.time() < deadline - 10:
            front_indices = []
            fronts = self._fast_non_dominated_sort(F)
            if fronts:
                front_indices = fronts[0]
            
            # Polish each point in the first front with a few local steps
            for idx in front_indices:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                x_best = X[idx].copy()
                f_best = F[idx].copy()
                # Simple local search with decreasing step size
                for step_mult in [0.05, 0.02, 0.01, 0.005]:
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    # Try perturbations in each dimension
                    for d in range(self.dim):
                        for direction in [-1, 1]:
                            x_trial = x_best.copy()
                            x_trial[d] += direction * step_mult
                            if np.any(x_trial < 0) or np.any(x_trial > 1):
                                continue
                            f_trial = problem(x_trial)
                            used += 1
                            # If it improves on f_best in at least one objective and doesn't worsen others too much
                            if np.all(f_trial <= f_best + 1e-6) and np.any(f_trial < f_best - 1e-6):
                                x_best = x_trial
                                f_best = f_trial
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                if used < self.budget:
                    all_X = np.vstack([all_X, x_best.reshape(1, -1)])
                    all_F = np.vstack([all_F, f_best.reshape(1, -1)])

        return all_X

    def _tournament_select(self, F, fronts):
        """Tournament selection based on front rank."""
        rank = np.zeros(len(F), dtype=np.int32)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i

        t1 = np.random.randint(len(F))
        t2 = np.random.randint(len(F))
        if rank[t1] < rank[t2]:
            return t1
        elif rank[t2] < rank[t1]:
            return t2
        else:
            return np.random.choice([t1, t2])
# EVOLVE-BLOCK-END
