"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol sampling and NSGA-II refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(points):
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(points[j], points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            if len(nd_idx) <= 2:
                return np.ones(len(nd_idx))
            
            dist = np.zeros(len(nd_idx))
            m = F.shape[1]  # number of objectives
            
            for k in range(m):
                sorted_idx = np.argsort(F[nd_idx, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F[nd_idx, k].max() - F[nd_idx, k].min()
                if f_range > 0:
                    for i in range(1, len(nd_idx) - 1):
                        dist[sorted_idx[i]] += (F[nd_idx, k][sorted_idx[i + 1]] - F[nd_idx, k][sorted_idx[i - 1]]) / f_range
            
            return dist

        # Phase 1: Global Sampling using Sobol sequence
        n_global = 100
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit = sampler.random(n_global)
        X_global = lo + points_unit * (hi - lo)
        
        F_global = np.array([problem(x) for x in X_global])
        used = n_global

        # Phase 2: NSGA-II style refinement
        # Population size for refinement
        pop_size = 30
        n_generations = (budget - used) // pop_size
        
        # Initialize population with best non-dominated points + random
        front_indices = non_dominated_front(F_global)
        
        # Select initial population: mix of non-dominated and random
        X_pop = np.zeros((pop_size, dim))
        F_pop = np.zeros((pop_size, self.n_obj))
        
        # Fill with non-dominated points first
        n_nd = min(len(front_indices), pop_size // 2)
        if n_nd > 0:
            # Sort by hypervolume contribution proxy (sum of objectives)
            front_points = F_global[front_indices]
            order = np.argsort(np.sum(front_points, axis=1))
            selected = front_indices[order[:n_nd]]
            X_pop[:n_nd] = X_global[selected]
            F_pop[:n_nd] = F_global[selected]
        
        # Fill rest with random points
        rng = np.random.default_rng(123)
        for i in range(n_nd, pop_size):
            if used >= budget or time.time() > deadline - 5:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            X_pop[i] = x
            F_pop[i] = f
            used += 1
        
        # NSGA-II loop
        for gen in range(n_generations):
            if used >= budget or time.time() > deadline - 5:
                break
            
            # Combine population and offspring (empty initially)
            # For simplicity, we'll just evolve the population in-place
            
            # Get non-dominated front
            nd_idx = non_dominated_front(F_pop)
            
            if len(nd_idx) == 0:
                # All dominated, pick the best single point
                best_idx = np.argmin(np.sum(F_pop, axis=1))
                nd_idx = np.array([best_idx])
            
            # Compute crowding distance for selection
            cd = crowding_distance(F_pop, nd_idx)
            
            # Select parents: prefer non-dominated with higher crowding distance
            # For each new child, select 2 parents via tournament selection
            for child_idx in range(pop_size):
                if used >= budget or time.time() > deadline - 5:
                    break
                
                # Tournament selection for parent 1
                p1_idx = self._tournament_select(F_pop, cd, nd_idx, 2, rng)
                # Tournament selection for parent 2
                p2_idx = self._tournament_select(F_pop, cd, nd_idx, 2, rng)
                
                # Crossover (SBX-like)
                if p1_idx != p2_idx:
                    alpha = rng.random()
                    child = alpha * X_pop[p1_idx] + (1 - alpha) * X_pop[p2_idx]
                else:
                    child = X_pop[p1_idx].copy()
                
                # Mutation
                step_size = 0.15 * (hi - lo)
                child = child + step_size * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                # Use a combined score: dominated points get high score, non-dominated get low
                # For simplicity, replace the point with highest sum of objectives among dominated
                # or lowest crowding distance among non-dominated
                
                # Simple approach: replace the worst by sum of objectives
                worst_idx = int(np.argmax(np.sum(F_pop, axis=1)))
                X_pop[worst_idx] = child
                F_pop[worst_idx] = fc
        
        # Combine all evaluated points
        # All points in X_pop and F_global have been evaluated
        # We need to return the non-dominated set from the union
        
        # Collect all unique points
        all_X = np.vstack([X_global, X_pop])
        all_F = np.vstack([F_global, F_pop])
        
        # Remove duplicates (approximately)
        # For simplicity, just take non-dominated from all_F
        final_front = non_dominated_front(all_F)
        
        return all_X[final_front]

    def _tournament_select(self, F, cd, nd_idx, k, rng):
        """Tournament selection."""
        n = len(F)
        # Prefer non-dominated points
        # Assign rank: 0 for non-dominated, 1 for dominated
        rank = np.ones(n)
        rank[nd_idx] = 0
        
        # Tournament
        candidates = rng.integers(0, n, size=k)
        # Sort by rank, then by crowding distance (higher is better for non-dominated)
        # For dominated, just use sum of objectives
        scores = []
        for c in candidates:
            if rank[c] == 0:
                # Non-dominated: higher crowding distance is better
                scores.append(-cd[np.where(nd_idx == c)[0][0]] if len(np.where(nd_idx == c)[0]) > 0 else 0)
            else:
                # Dominated: lower sum of objectives is better
                scores.append(np.sum(F[c]))
        
        best = candidates[np.argmin(scores)]
        return best
# EVOLVE-BLOCK-END
