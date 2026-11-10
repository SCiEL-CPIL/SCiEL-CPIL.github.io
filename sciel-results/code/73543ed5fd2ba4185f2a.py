"""Multi-objective optimizer for ZDT3 using NSGA-II evolutionary search."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if solution a dominates solution b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Perform fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
        n_count = [0] * n           # n_count[i] = number of solutions dominating i
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_count[i] += 1
            if n_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove last empty front

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        
        cd = [0.0] * n
        for m in range(self.n_obj):
            obj_vals = np.array([F[idx][m] for idx in front_indices])
            sorted_idx = np.argsort(obj_vals)
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
            if obj_range < 1e-15:
                continue
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / obj_range
        
        return cd

    def _selection_tournament(self, rank, cd, size):
        """Tournament selection based on rank and crowding distance."""
        n = len(rank)
        winners = []
        for _ in range(size):
            i, j = np.random.randint(0, n, 2)
            if rank[i] != rank[j]:
                if rank[i] < rank[j]:
                    winners.append(i)
                else:
                    winners.append(j)
            else:
                if cd[i] > cd[j]:
                    winners.append(i)
                else:
                    winners.append(j)
        return winners

    def _crossover(self, x1, x2):
        """Simulated binary crossover."""
        if np.random.rand() < 0.9:
            eta = 20
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x1)
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        r = np.random.rand()
                        if r <= 0.5:
                            beta = (2 * r) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - r))) ** (1 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2
        else:
            return x1.copy(), x2.copy()

    def _mutation(self, x, lo, hi):
        """Polynomial mutation."""
        eta = 20
        x_mut = x.copy()
        for i in range(len(x_mut)):
            if np.random.rand() < 0.9:
                range_val = hi[i] - lo[i]
                if range_val < 1e-14:
                    continue
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))
                x_mut[i] += delta * range_val
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # NSGA-II parameters
        pop_size = 100
        max_generations = self.budget // pop_size  # 2 generations for budget=200
        # Actually, we need to be more careful. Let's use a smaller population
        # and more generations, or just do one big generation.
        
        # Let's use pop_size=50, and do 4 generations (50*4=200 evaluations)
        pop_size = 50
        max_generations = self.budget // pop_size
        
        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        
        # Evaluate initial population
        F = np.zeros((pop_size, self.n_obj))
        evals_used = 0
        for i in range(pop_size):
            if time.time() > deadline:
                break
            try:
                x = np.clip(X[i], lo, hi)
                F[i] = problem(x)
                evals_used += 1
            except Exception:
                F[i] = [1e10, 1e10]
        
        # If we didn't finish initial population, reduce pop_size
        if evals_used < pop_size:
            pop_size = evals_used
            X = X[:pop_size]
            F = F[:pop_size]
        
        # Keep track of all evaluated points for final result
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Evolution loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(pop_size, dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx
            
            # Compute crowding distances
            cd = np.zeros(pop_size)
            for front in fronts:
                front_cd = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Create mating pool via tournament selection
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                if time.time() > deadline:
                    break
                parent1_idx = self._selection_tournament(rank, cd, 2)
                parent2_idx = self._selection_tournament(rank, cd, 2)
                p1 = X[parent1_idx[0]]
                p2 = X[parent2_idx[0]]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutation(c1, lo, hi)
                c2 = self._mutation(c2, lo, hi)
                if i < pop_size:
                    offspring_X[i] = c1
                if i + 1 < pop_size:
                    offspring_X[i + 1] = c2
            
            # Evaluate offspring
            offspring_F = np.zeros((pop_size, self.n_obj))
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                try:
                    x = np.clip(offspring_X[i], lo, hi)
                    offspring_F[i] = problem(x)
                    evals_used += 1
                except Exception:
                    offspring_F[i] = [1e10, 1e10]
            
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined population
            combined_fronts = self._non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            new_rank = np.zeros(pop_size, dtype=int)
            new_cd = np.zeros(pop_size)
            
            count = 0
            for front_idx, front in enumerate(combined_fronts):
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        new_rank[count] = front_idx
                        count += 1
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - count
                    front_cd = self._crowding_distance(combined_F, front)
                    # Sort front by crowding distance descending
                    front_sorted = sorted(range(len(front)), key=lambda j: -front_cd[j])
                    for j in front_sorted[:remaining]:
                        idx = front[j]
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        new_rank[count] = front_idx
                        count += 1
                    break
            
            X = new_X[:count]
            F = new_F[:count]
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        # Return the last population (or we could return all, but the interface expects a single return)
        # The evaluator records all objective vectors from problem() calls, so we just need to make
        # sure we've made the calls. The return value is ignored.
        return X

# EVOLVE-BLOCK-END
