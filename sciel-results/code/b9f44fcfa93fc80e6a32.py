"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: 40 is a good balance for 200 budget
        # 40 initial + 4 generations of 40 offspring = 200 evals
        pop_size = 40
        
        if self.budget < pop_size:
            pop_size = self.budget

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns a list of lists, where each inner list contains indices of individuals
            belonging to that front (front 0, front 1, etc.).
            """
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]

            # Vectorized dominance check is tricky for full N^2 in pure python loops without overhead,
            # but for N=40, even a loop is fast. We use a loop for clarity and correctness.
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1

                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts[:-1]

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a front.
            """
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                if obj_vals.max() == obj_vals.min():
                    continue
                range_val = obj_vals.max() - obj_vals.min()
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / range_val
            return dist

        def tournament_selection(X, F):
            """
            Binary tournament selection based on non-dominated rank and crowding distance.
            Returns index of the selected parent.
            """
            # Compute ranks and crowding distances for the current population
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Crowding distance
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]

            # Binary tournament
            i1, i2 = rng.integers(0, len(F), size=2)
            if rank[i1] < rank[i2]:
                return i1
            elif rank[i2] < rank[i1]:
                return i2
            else:
                return i1 if crowd[i1] > crowd[i2] else i2

        def sbx_crossover(p1, p2):
            """
            Simulated Binary Crossover.
            """
            eta = 20.0
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    child1[d] = p1[d]
                    child2[d] = p2[d]
                else:
                    if p1[d] == p2[d]:
                        child1[d] = p1[d]
                        child2[d] = p2[d]
                    else:
                        if p1[d] < p2[d]:
                            x1, x2 = p1[d], p2[d]
                        else:
                            x1, x2 = p2[d], p1[d]
                        
                        # Calculate beta
                        if x2 > x1:
                            beta = 1.0 + (2.0 * x1 / (x2 - x1))
                        else:
                            beta = 1.0 + (2.0 * x2 / (x1 - x2))
                        
                        alpha1 = (beta + 1.0) / (eta + 1.0)
                        alpha2 = (beta + 1.0) / (eta + 1.0)
                        
                        rand1 = rng.random()
                        rand2 = rng.random()
                        
                        if rand1 <= 1.0 / alpha1:
                            beta_q1 = (1.0 / alpha1) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q1 = (2.0 - rand1 * (eta + 1.0)) ** (-1.0 / eta)
                            
                        if rand2 <= 1.0 / alpha2:
                            beta_q2 = (1.0 / alpha2) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q2 = (2.0 - rand2 * (eta + 1.0)) ** (-1.0 / eta)

                        c1 = 0.5 * ((x1 + x2) - (x2 - x1) * beta_q1)
                        c2 = 0.5 * ((x1 + x2) + (x2 - x1) * beta_q2)
                        
                        child1[d] = c1
                        child2[d] = c2
            return child1, child2

        def polynomial_mutation(x):
            """
            Polynomial mutation.
            """
            eta = 20.0
            y = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.5:
                    continue
                delta1 = (x[d] - lo[d]) / (hi[d] - lo[d])
                delta2 = (hi[d] - x[d]) / (hi[d] - lo[d])
                rand = rng.random()
                
                if rand < 0.5:
                    xy = 1.0 + (2.0 * (1.0 - delta2))
                    beta = 1.0 + (2.0 * (1.0 - delta1))
                    mut = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = (beta ** (-1.0 / eta)) if rand <= 0.5 else (2.0 - mut) ** (-1.0 / eta)
                else:
                    xy = 1.0 + (2.0 * delta1)
                    beta = 1.0 + (2.0 * delta2)
                    mut = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = (beta ** (-1.0 / eta)) if rand > 0.5 else (2.0 - mut) ** (-1.0 / eta)
                
                y[d] = x[d] + delta_q * (hi[d] - lo[d])
            return np.clip(y, lo, hi)

        # Main loop
        while evals < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                p1_idx = tournament_selection(X, F)
                p2_idx = tournament_selection(X, F)
                
                if p1_idx == p2_idx:
                    child = polynomial_mutation(X[p1_idx].copy())
                else:
                    c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                    child = polynomial_mutation(c1.copy())
                
                offspring_X.append(child)

            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_all)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_all, front)
                    # Sort indices in front by crowding distance descending
                    sorted_indices = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

# EVOLVE-BLOCK-END
