"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4: first dim-5 variables shape the front, last 5 affect g
        n_front = dim - 5  # 7 variables for front
        n_g = 5            # 5 variables for g

        # Population size
        pop_size = min(100, max(20, budget // 2))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Initialize population near the Pareto front
        # For DTLZ4, the Pareto front is:
        # f1 = (1+g)*0.5*cos(pi*x1/2)*cos(pi*x2/2)
        # f2 = (1+g)*0.5*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = (1+g)*0.5*sin(pi*x1/2)
        # with g = sum((x_i - 0.5)^2) for i in 8..12 (1-indexed)
        # To get g=0, set x8..x12 = 0.5
        # For the front shape, x1..x7 should be in [0,1]

        # Create initial population with good front coverage
        # Use spherical parameterization for the front
        theta1 = np.linspace(0, np.pi / 2, pop_size, endpoint=False)
        theta2 = np.linspace(0, np.pi, pop_size, endpoint=False)

        # For DTLZ4 with n_obj=3, only x1 and x2 matter for the front
        # x3..x7 can be anything (they don't affect the front in standard DTLZ4)
        # Actually, in DTLZ4, all first dim-5 variables can affect the front
        # But typically only the first n_obj-1 variables are used for the front

        # Let's use a simpler approach: random initialization with bias toward good regions
        # Set g-variables (last 5) near 0.5 to keep g small
        X = np.zeros((pop_size, dim))
        # Front variables: uniform in [0,1]
        X[:, :n_front] = np.random.rand(pop_size, n_front)
        # G variables: near 0.5 (which gives g=0)
        X[:, n_front:] = 0.5 + 0.05 * np.random.randn(pop_size, n_g)
        X = np.clip(X, lo, hi)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sorting
        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
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

            # Add any remaining (shouldn't happen, but safety)
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)

            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            m = len(indices)
            n_obj = F.shape[1]
            dist = np.zeros(m)

            for k in range(n_obj):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return dist

        # Selection using tournament
        def tournament_select(fronts, crowd_dists, X, F, k=2):
            """Select k parents via tournament."""
            # Create a rank array
            n = len(F)
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            parents = []
            for _ in range(k):
                candidates = np.random.choice(n, size=k, replace=False)
                best = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best]:
                        # Use crowding distance (higher is better)
                        # Find crowding distance for these indices
                        # This is expensive, so use a simpler heuristic
                        # For now, just pick randomly among tied
                        if np.random.rand() < 0.5:
                            best = c
                parents.append(X[best])
            return parents

        # Mutation
        def mutate(parent, sigma):
            """Gaussian mutation with clipping."""
            child = parent + sigma * np.random.randn(dim)
            return np.clip(child, lo, hi)

        # Crossover (SBX-like, simplified)
        def crossover(p1, p2):
            """Simple blend crossover."""
            if np.random.rand() < 0.5:
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = (1 - alpha) * p1 + alpha * p2
                return np.clip(c1, lo, hi), np.clip(c2, lo, hi)
            return p1.copy(), p2.copy()

        # Main loop
        while used < budget:
            if time.time() - t0 > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Compute crowding distances for the first front (most important)
            # For efficiency, only compute for the first few fronts
            crowd_dists = np.zeros(len(F))
            for front in fronts[:3]:  # Only first 3 fronts
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowd_dists[idx] = cd[i]

            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            offspring_X = []
            for _ in range(n_offspring):
                if time.time() - t0 > deadline:
                    break

                # Select parents
                parent1, parent2 = tournament_select(fronts, crowd_dists, X, F, k=2)

                # Crossover
                c1, c2 = crossover(parent1, parent2)

                # Mutation - adaptive sigma
                # Reduce sigma as we converge
                progress = used / budget
                sigma = 0.2 * (1 - 0.5 * progress)

                c1 = mutate(c1, sigma)
                c2 = mutate(c2, sigma)

                offspring_X.append(c1)
                offspring_X.append(c2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:n_offspring])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)

            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort the combined population
            combined_fronts = non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.append(combined_X[front])
                    next_F.append(combined_F[front])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    sorted_idx = np.argsort(cd)[::-1]
                    selected = [front[i] for i in sorted_idx[:remaining]]
                    next_X.append(combined_X[selected])
                    next_F.append(combined_F[selected])
                    break

            X = np.vstack(next_X)
            F = np.vstack(next_F)

        return X
# EVOLVE-BLOCK-END
