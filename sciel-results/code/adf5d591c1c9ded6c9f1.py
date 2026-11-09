"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices) and rank array."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        rank = np.full(n, -1, dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[j] += 1

        current_front = []
        for i in range(n):
            if n_dom[i] == 0:
                current_front.append(i)
                rank[i] = 0

        if current_front:
            fronts.append(current_front)

        k = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
                        rank[j] = k + 1
            k += 1
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts, rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        m = self.n_obj
        for k in range(m):
            vals = F[indices, k]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            span = vals[order[-1]] - vals[order[0]]
            if span > 1e-15:
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / span
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(50, max(20, budget // 4))

        # Random initialization
        X = lo + span * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts, rank = self._fast_non_dominated_sort(F)

            # Build combined population for selection
            # Use tournament selection with rank + crowding distance
            # Compute crowding distance for each front
            cd_by_front = []
            for front in fronts:
                if len(front) > 2:
                    cd = self._crowding_distance(F, front)
                else:
                    cd = np.full(len(front), np.inf)
                cd_by_front.append((front, cd))

            # Create a combined list for tournament
            # Assign fitness: lower rank is better, then higher crowding is better
            # We'll use a simple approach: select parents via tournament on (rank, -cd)

            # Build parent pool
            parent_pool = []
            for fi, (front, cd) in enumerate(cd_by_front):
                for idx, cdi in zip(front, cd):
                    parent_pool.append((fi, -cdi, idx))

            # Tournament selection
            def tournament_select(pool):
                k = 3
                candidates = [pool[np.random.randint(len(pool))] for _ in range(k)]
                best = min(candidates, key=lambda x: (x[0], x[1]))
                return best[2]

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = []
            children_F = []

            for _ in range(n_children):
                p1 = tournament_select(parent_pool)
                p2 = tournament_select(parent_pool)

                # SBX-like crossover (blend)
                if np.random.rand() < 0.9:
                    # Simulated binary crossover
                    beta = 0.5
                    r1 = np.random.rand(dim)
                    r2 = np.random.rand(dim)
                    x1 = X[p1]
                    x2 = X[p2]
                    # Blend crossover
                    child1 = (1 - beta) * x1 + beta * x2
                    child2 = (1 - beta) * x2 + beta * x1
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Mutation: polynomial-like or Gaussian
                # Use Gaussian mutation with adaptive step
                step = 0.1 * span
                child1 = child1 + step * np.random.randn(dim)
                child2 = child2 + step * np.random.randn(dim)

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                children_X.append(child1)
                children_F.append(problem(child1))
                used += 1

                if used >= budget:
                    break

                if time.time() > deadline:
                    break

                if np.random.rand() < 0.5:
                    children_X.append(child2)
                    children_F.append(problem(child2))
                    used += 1
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])

            # Non-dominated sort on combined
            fronts, rank = self._fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to trim this front by crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # descending crowding
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
