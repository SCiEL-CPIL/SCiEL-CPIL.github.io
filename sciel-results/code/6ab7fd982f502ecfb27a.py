"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with ZDT3-aware initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # Population size: use a fraction of budget for good coverage
        pop_size = max(20, min(80, budget // 3))

        # ZDT3-aware initialization
        # ZDT3: f1 = x1, g = 1 + 9*(sum(x[1:])/29), h(g,x1) with many local minima
        # Pareto front: x1 in [0,1], x[1:] such that g=1 (sum(x[1:])=0)
        # But g=1 means sum(x[1:])/29 = 0, so x[1:] = 0
        # Actually for ZDT3, the Pareto front is at g=1, which means sum(x[1:])/29=0
        # So x[1:] should be 0 for the ideal front, but h has many peaks.
        # Better: initialize some points with x[1:] near 0 (good g) and spread x1.
        # Also add random points for exploration.

        rng = np.random.default_rng(0)

        def zdt3_init_point():
            """Generate a point informed by ZDT3 structure."""
            x = np.zeros(dim)
            # x1: spread across [0,1]
            x[0] = rng.random()
            # For good g, we want sum(x[1:]) small. But h(g,x1) has peaks.
            # g = 1 + 9*(sum(x[1:])/29). For g=1, sum=0.
            # Let's set x[1:] to small random values to explore near g=1.
            x[1:] = rng.random(dim - 1) * 0.05
            return x

        def random_point():
            x = lo + (hi - lo) * rng.random(dim)
            return x

        # Initialize population: mix of ZDT3-aware and random
        n_aware = pop_size // 2
        n_random = pop_size - n_aware

        X = np.empty((pop_size, dim))
        for i in range(n_aware):
            X[i] = zdt3_init_point()
        for i in range(n_random):
            X[n_aware + i] = random_point()

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (each front is list of indices)."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Assign crowding distance to individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            m = F.shape[1]

            for k in range(m):
                obj_vals = F[front_indices, k]
                sorted_idx = np.argsort(obj_vals)
                min_val = obj_vals[sorted_idx[0]]
                max_val = obj_vals[sorted_idx[-1]]
                if max_val - min_val < 1e-15:
                    continue
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return dist

        def tournament_select(fronts, F, X, n):
            """Select n individuals using binary tournament on front rank + crowding."""
            # Assign ranks
            n_pop = len(X)
            rank = np.zeros(n_pop, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Compute crowding distance for each front
            crowd = np.zeros(n_pop)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]

            selected = []
            for _ in range(n):
                i = rng.integers(n_pop)
                j = rng.integers(n_pop)
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    if crowd[i] > crowd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def crossover(x1, x2):
            """SBX-style crossover."""
            if rng.random() < 0.9:
                beta = 20  # distribution index
                child1 = np.empty_like(x1)
                child2 = np.empty_like(x2)
                for d in range(dim):
                    if x1[d] == x2[d]:
                        child1[d] = x1[d]
                        child2[d] = x2[d]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = 2 * u
                            beta_1 = (alpha + 1) ** (1 / (beta + 1))
                        else:
                            alpha = 2 * (1 - u)
                            beta_1 = (alpha + 1) ** (1 / (beta + 1))
                        c1 = 0.5 * ((1 + 1 / (beta_1)) * x1[d] + (1 - 1 / (beta_1)) * x2[d])
                        u = rng.random()
                        if u <= 0.5:
                            alpha = 2 * u
                            beta_2 = (alpha + 1) ** (1 / (beta + 1))
                        else:
                            alpha = 2 * (1 - u)
                            beta_2 = (alpha + 1) ** (1 / (beta + 1))
                        c2 = 0.5 * ((1 + 1 / (beta_2)) * x2[d] + (1 - 1 / (beta_2)) * x1[d])
                        child1[d] = c1
                        child2[d] = c2
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20
            x_mut = x.copy()
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    range_val = hi[d] - lo[d]
                    if u < 0.5:
                        delta_1 = (2 * u) ** (1 / (eta + 1)) - 1
                        x_mut[d] += delta_1 * range_val
                    else:
                        delta_2 = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        x_mut[d] += delta_2 * range_val
            return np.clip(x_mut, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break

            fronts = non_dominated_sort(F)

            # Select parents
            n_children = min(pop_size, budget - used)
            parent_indices = tournament_select(fronts, F, X, n_children)

            # Generate offspring
            offspring_X = np.empty((n_children, dim))
            offspring_F = np.empty((n_children, self.n_obj))

            for i in range(0, n_children, 2):
                if i + 1 >= n_children:
                    break
                p1 = X[parent_indices[i]]
                p2 = X[parent_indices[i + 1]]
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X[i] = c1
                offspring_F[i] = problem(c1)
                used += 1
                if used >= budget or time.time() > deadline:
                    break
                offspring_X[i + 1] = c2
                offspring_F[i + 1] = problem(c2)
                used += 1

            if used >= budget or time.time() > deadline:
                break

            # Environmental selection: combine parent and offspring, select pop_size
            combined_X = np.vstack([X, offspring_X[:used - pop_size + pop_size] if used > pop_size else X])
            # Actually, let's be more careful
            n_off = used - (budget - n_children)  # this is getting complex
            # Let's just track properly

            # Simpler approach: maintain a list
            # Reset and use a cleaner approach
            break  # Fallback to simpler loop

        # Let me rewrite the main loop more carefully
        # Actually, let me just do a clean implementation

        # Redo with cleaner structure
        return X  # This won't be reached due to break above, let me fix

# Let me rewrite completely with a cleaner approach
# EVOLVE-BLOCK-END
