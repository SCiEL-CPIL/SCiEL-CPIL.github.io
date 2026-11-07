"""Multi-objective optimizer on DTLZ7 using a population-based evolutionary algorithm with crowding distance."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using NSGA-II-like approach:
    - Fast non-dominated sorting
    - Crowding distance for diversity
    - Tournament selection
    - Mutation and crossover
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Track evaluations
        eval_count = 0
        all_points = []
        all_objectives = []
        
        def eval_point(x):
            nonlocal eval_count
            eval_count += 1
            f = problem(x)
            all_points.append(x.copy())
            all_objectives.append(np.array(f))
            return np.array(f)
        
        # Population size
        pop_size = min(30, max(10, self.budget // 10))
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([eval_point(x) for x in X])
        
        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns list of fronts, each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = [0] * n  # n_dominates[i] = number of points dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            
            return fronts
        
        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices.
            Returns array of crowding distances.
            """
            if len(indices) <= 2:
                return np.ones(len(indices))
            
            n = len(indices)
            cd = np.zeros(n)
            n_obj = F.shape[1]
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[indices[i], m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[indices[sorted_idx[-1]], m] - F[indices[sorted_idx[0]], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]], m] - F[indices[sorted_idx[i - 1]], m]) / f_range
            
            return cd
        
        def tournament_select(X, F, fronts, k=2):
            """
            Tournament selection.
            Returns index of selected individual.
            """
            n = len(F)
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            # Compare by front rank, then crowding distance
            front_rank = {idx: rank for rank, front in enumerate(fronts) for idx in front}
            # Calculate crowding distance for each front
            cd_dict = {}
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_dict[idx] = cd[i]
            
            for c in candidates[1:]:
                if front_rank.get(best, len(fronts)) > front_rank.get(c, len(fronts)):
                    best = c
                elif front_rank.get(best, len(fronts)) == front_rank.get(c, len(fronts)):
                    if cd_dict.get(best, 0) < cd_dict.get(c, 0):
                        best = c
            return best
        
        def crossover(x1, x2):
            """Simulated binary crossover (simplified)."""
            x1 = x1.copy()
            x2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    # Swap
                    x1[i], x2[i] = x2[i], x1[i]
            return x1, x2
        
        def mutate(x, rate=0.1, sigma=0.1):
            """Gaussian mutation."""
            x = x.copy()
            mask = rng.random(self.dim) < rate
            x[mask] += sigma * span[mask] * rng.standard_normal(np.sum(mask))
            return np.clip(x, lo, hi)
        
        # Evolution loop
        while eval_count < self.budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and eval_count < self.budget:
                # Select parents
                fronts = fast_non_dominated_sort(F)
                p1 = tournament_select(X, F, fronts)
                p2 = tournament_select(X, F, fronts)
                
                # Crossover
                c1, c2 = crossover(X[p1], X[p2])
                
                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                # Evaluate
                f1 = eval_point(c1)
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if eval_count < self.budget:
                    f2 = eval_point(c2)
                    offspring_X.append(c2)
                    offspring_F.append(f2)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Select next generation
            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd = crowding_distance(combined_F, front)
                    remaining = pop_size - len(new_X)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return all_points
# EVOLVE-BLOCK-END
