import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        # WFG9 specific: We know the Pareto front structure.
        # However, we must search. A strong approach for WFG9 is to use
        # a hybrid: initial random, then guided local search or better EA.
        # Given the budget is 200, which is very small for 12 dims,
        # we need a very efficient algorithm.
        # NSGA-II is standard but might be slow in Python for 12 dims if not vectorized well.
        # Let's try a simpler but effective approach:
        # 1. Initialize with a mix of random and structured points.
        # 2. Use a fast non-dominated sort.
        # 3. Use a compact genetic algorithm or just random perturbation of best points.

        # Given the small budget (200), a population-based EA might be overkill if the overhead is high.
        # Let's try a "Greedy Local Search" hybrid:
        # 1. Start with a small diverse population.
        # 2. Iteratively pick the best point (non-dominated, high crowding) and perturb it.
        # 3. Keep the non-dominated set.
        
        # Actually, for WFG9, the Pareto front is well-known.
        # The objectives are f1, f2, f3.
        # f1 = x1
        # f2 = (1 - x1) * (1 + 0.1 * (f1 + f3)) ... wait, standard WFG9:
        # f1 = x1
        # f2 = (1 - x1) * (1 + 0.1 * (f1 + f3))
        # f3 = (1 - x1) * (1 + 0.1 * (f1 + f2))
        # This is not quite right.
        
        # Let's stick to a robust EA but optimized for speed.
        # The key is to minimize overhead.
        
        # Population size
        N = 20
        
        # Initial population: random + some structured
        # Try to spread out in the objective space by using a grid in decision space?
        # Or just random.
        
        X = lo + np.random.rand(N, self.dim) * (hi - lo)
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: fast non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
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
            if current_front:
                fronts.append(current_front)
            
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
            return fronts

        # Main loop
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            # Get fronts
            fronts = non_dominated_sort(F)
            
            # Select best individual from first front (or first few)
            # To maintain diversity, we can pick from the first front with some probability,
            # or use crowding distance.
            
            # For simplicity and speed, let's pick the individual with the highest crowding distance
            # from the first front. If the first front has only 1, pick it.
            
            # Calculate crowding distance for the first front
            front0 = fronts[0]
            if len(front0) == 1:
                best_idx = front0[0]
            else:
                # Simple crowding: max sum of differences to neighbors in each objective
                # Or just pick randomly from front0 to avoid bias?
                # Let's pick the one with the "best" average objective value?
                # No, that converges to one point.
                # Let's use a simple diversity measure: distance to centroid of front0
                centroid = np.mean(F[front0], axis=0)
                dists = np.linalg.norm(F[front0] - centroid, axis=1)
                best_idx = front0[np.argmax(dists)]
            
            x_parent = X[best_idx]
            
            # Generate offspring by perturbing the parent
            # Use a Gaussian perturbation scaled by the span
            # Adaptive step size?
            step_size = 0.1 * span
            x_child = x_parent + np.random.normal(0, step_size)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            # Add to population
            X = np.vstack([X, x_child])
            F = np.vstack([F, f_child])
            
            # If population grows too large, prune it
            # Keep non-dominated solutions. If there are too many, keep the most diverse.
            if len(X) > 50:  # Threshold
                fronts = non_dominated_sort(F)
                # Keep first two fronts
                keep_indices = []
                for front in fronts[:2]:
                    keep_indices.extend(front)
                
                if len(keep_indices) > N:
                    # If still too many, reduce to N
                    # Keep the most diverse ones
                    # Simple method: random sample
                    keep_indices = np.random.choice(keep_indices, N, replace=False)
                
                X = X[keep_indices]
                F = F[keep_indices]
        
        return X
