"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using a population-based approach
    with crowding distance and fast non-dominated sorting (NSGA-II style),
    plus a deadline guard for safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(92, max(20, budget // 2))
        
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        mutation_rate = 0.8
        crossover_rate = 0.9

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # If some points are not assigned (shouldn't happen), add them to last front
            assigned = set()
            for front in fronts:
                assigned.update(front)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                if fronts:
                    fronts[-1].extend(remaining)
                else:
                    fronts.append(remaining)
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front. Returns array of distances."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-12:
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return dist

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Create next generation
            children = []
            n_children = pop_size

            for _ in range(n_children):
                # Tournament selection
                if len(children) == 0 and rng.random() < crossover_rate and pop_size >= 2:
                    # Binary tournament
                    i1, i2 = rng.integers(0, pop_size, size=2)
                    if ranks[i1] < ranks[i2]:
                        p1 = i1
                    elif ranks[i2] < ranks[i1]:
                        p1 = i2
                    else:
                        p1 = i1 if rng.random() < 0.5 else i2
                    
                    i1, i2 = rng.integers(0, pop_size, size=2)
                    if ranks[i1] < ranks[i2]:
                        p2 = i1
                    elif ranks[i2] < ranks[i1]:
                        p2 = i2
                    else:
                        p2 = i1 if rng.random() < 0.5 else i2
                    
                    # SBX-like crossover
                    child = (X[p1] + X[p2]) / 2.0
                    child += 0.05 * (hi - lo) * rng.standard_normal(dim)
                else:
                    # Just pick a random parent and mutate
                    p = rng.integers(0, pop_size)
                    child = X[p].copy()
                    for d in range(dim):
                        if rng.random() < mutation_rate:
                            child[d] += 0.05 * (hi - lo) * rng.standard_normal()
                
                child = np.clip(child, lo, hi)
                children.append(child)

            children = np.array(children)
            F_children = np.array([problem(x) for x in children])
            used += len(children)

            # Combine parents and children
            X_all = np.vstack([X, children])
            F_all = np.vstack([F, F_children])

            # Fast non-dominated sort on combined population
            fronts_all = fast_non_dominated_sort(F_all)

            # Select next population
            X_new = []
            F_new = []
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(X_new)
                    # Compute crowding distance
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return X
# EVOLVE-BLOCK-END
