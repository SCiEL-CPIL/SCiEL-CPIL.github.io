# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 (12 vars, 3 objs).
    Uses NSGA-II framework with Differential Evolution (DE) style mutation
    and structured initialization near the known Pareto front region.
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
        rng = np.random.default_rng(42)
        
        # Population size
        # Smaller population allows more generations within the budget,
        # which is often better for convergence in low-budget scenarios.
        pop_size = 30
        if pop_size > budget:
            pop_size = budget
            
        # Initialize population:
        # WFG9 PF is typically found where variables are small.
        # Use a mix of random and "near-zero" initialization.
        X = np.zeros((pop_size, dim))
        
        # 30% random, 70% near origin (weighted towards smaller values)
        n_random = pop_size // 3
        n_origin = pop_size - n_random
        
        # Random
        X[:n_random] = lo + (hi - lo) * rng.random((n_random, dim))
        
        # Near origin: Sample from [lo, lo + 0.5*(hi-lo)] to cover the likely PF region better
        # WFG9 is often scaled such that [0,1] or similar is relevant.
        # Assuming standard WFG9 bounds are [0,1] or [-1,1], "near origin" means small absolute values.
        # We sample from a range that is 50% of the total range starting from the lower bound.
        # If bounds are symmetric, this might bias towards one side, but WFG9 PF is often in the first quadrant of objective space,
        # corresponding to small x values in many formulations.
        # Let's use a uniform distribution in the lower half of the box.
        range_50 = 0.5 * (hi - lo)
        X[n_random:] = lo + range_50 * rng.random((n_origin, dim))
        
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
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
                
                # DE-style mutation: v = x_p1 + F * (x_r2 - x_r3)
                # Select two more random parents for the differential part
                r2 = rng.integers(pop_size)
                r3 = rng.integers(pop_size)
                
                # DE factor F
                F_de = 0.5 + 0.5 * rng.random()
                
                # Create mutant vector
                mutant = X[p1] + F_de * (X[r2] - X[r3])
                
                # Add a small Gaussian perturbation for robustness
                # Step size decreases over time
                progress = used / budget
                step = 0.05 * (hi - lo) * (1.0 - 0.5 * progress)
                mutant = mutant + step * rng.standard_normal(dim)
                
                # Clip to bounds
                mutant = np.clip(mutant, lo, hi)
                
                # Evaluate
                fc = problem(mutant)
                offspring_X.append(mutant)
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
