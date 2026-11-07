# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 (12 vars, 3 objs).
    Combines structured random search (inspiration 2) with evolutionary refinement
    and adaptive mutation (inspiration 1/current program).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """
        Fast non-dominated sort.
        Returns:
            rank: array of rank for each point (0 is best)
            fronts: list of lists of indices, sorted by rank
        """
        n = len(F)
        if n == 0:
            return np.array([], dtype=int), []
        
        S = [[] for _ in range(n)]  # S[i] = list of indices dominated by i
        nd = np.zeros(n, dtype=int) # nd[i] = number of points dominating i
        fronts = []

        # Compute dominance relationships
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1

        rank = np.full(n, -1, dtype=int)
        
        # Find first front (non-dominated points)
        current_front = [i for i in range(n) if nd[i] == 0]
        front_idx = 0
        
        while current_front:
            next_front = []
            for i in current_front:
                rank[i] = front_idx
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            fronts.append(current_front)
            current_front = next_front
            front_idx += 1

        # Assign remaining (should not happen for standard NSGA-II, but safe)
        for i in range(n):
            if rank[i] == -1:
                rank[i] = front_idx
                if not fronts:
                    fronts.append([i])
                else:
                    fronts[-1].append(i)

        return rank, fronts

    def _crowding_distance(self, F, front):
        """
        Compute crowding distance for indices in a front.
        Returns array of distances for the points in 'front'.
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        Fsub = F[front]
        
        for m in range(Fsub.shape[1]):
            fvals = Fsub[:, m]
            fmin = fvals.min()
            fmax = fvals.max()
            
            if fmax == fmin:
                continue
            
            order = np.argsort(fvals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            
            for i in range(1, n - 1):
                idx_prev = order[i - 1]
                idx_next = order[i + 1]
                cd[idx_prev] += (fvals[idx_next] - fvals[idx_prev]) / (fmax - fmin)
        
        return cd

    def _select_next_population(self, F_all, X_all, pop_size):
        """
        Select next population using non-dominated sorting and crowding distance.
        """
        rank, fronts = self._non_dominated_sort(F_all)
        
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                need = pop_size - len(selected)
                cd_front = self._crowding_distance(F_all, front)
                order = np.argsort(-cd_front)
                for i in range(need):
                    selected.append(front[order[i]])
                break
        
        # If we still don't have enough (edge case), fill randomly
        if len(selected) < pop_size:
            remaining = [i for i in range(len(X_all)) if i not in selected]
            rng = np.random.default_rng(0)
            for _ in range(pop_size - len(selected)):
                if remaining:
                    idx = rng.integers(len(remaining))
                    selected.append(remaining.pop(idx))
        
        return selected

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 325  # 5s margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for internal RNG to ensure reproducibility independent of external seed
        # However, the task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # It's safer to use np.random directly or a default_rng that doesn't interfere.
        # Using default_rng(42) is good for internal consistency, but let's check if we should use global state.
        # The current program uses default_rng(42). Let's stick to that for stability.
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = 40
        if pop_size > budget:
            pop_size = budget
            
        # Initialize population:
        # Use a mix of random points and points near the origin (WFG9 PF is near origin)
        X = np.zeros((pop_size, dim))
        
        # 50% random, 50% near origin (small values)
        n_random = pop_size // 2
        n_origin = pop_size - n_random
        
        # Random
        X[:n_random] = lo + (hi - lo) * rng.random((n_random, dim))
        
        # Near origin (centered at 0, small range)
        # WFG9 variables are typically in [0, 1] or similar, but bounds are given.
        # Let's sample from a Gaussian centered at 0, clipped to bounds.
        # Or just uniform in [lo, lo + 0.2*(hi-lo)]
        range_20 = 0.2 * (hi - lo)
        X[n_random:] = lo + range_20 * rng.random((n_origin, dim))
        
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # Track best points found so far for return? No, we return X at the end.
        # But we should maintain a population that represents the current state of search.
        
        # Adaptive parameters
        # Mutation step size
        base_step = 0.15
        # Crossover rate
        cx_rate = 0.8
        
        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Compute ranks and crowding distance for current population
            rank, fronts = self._non_dominated_sort(F)
            
            cd_total = np.zeros(pop_size)
            for front in fronts:
                cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd_total[idx] = cd[i]
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Generate pop_size offspring
            for _ in range(pop_size):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                # Tournament selection for parent 1
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                def better(a, b):
                    if rank[a] != rank[b]:
                        return rank[a] < rank[b]
                    return cd_total[a] >= cd_total[b]
                p1 = i1 if better(i1, i2) else i2
                
                # Tournament selection for parent 2
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                p2 = i1 if better(i1, i2) else i2
                
                # Crossover or Mutation
                if rng.random() < cx_rate:
                    # Blend crossover
                    alpha = 0.3 + 0.4 * rng.random()
                    child = alpha * X[p1] + (1.0 - alpha) * X[p2]
                else:
                    # Clone
                    child = X[p1].copy()
                
                # Adaptive mutation
                # Reduce step size as we get closer to deadline or if we are stagnating?
                # Simple decay: step = base_step * (1 - 0.5 * (used/budget))
                progress = used / budget
                step = base_step * (1.0 - 0.3 * progress)
                
                # Gaussian mutation
                child = child + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                offspring_X.append(child)
                offspring_F.append(fc)
                used += 1
            
            if not offspring_X:
                break
            
            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])
            
            # Select next population
            selected = self._select_next_population(F_all, X_all, pop_size)
            
            X = X_all[selected]
            F = F_all[selected]
            
            if used >= budget:
                break
        
        return X

# EVOLVE-BLOCK-END
