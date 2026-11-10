import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 10
        # We can afford (budget - pop_size) / pop_size generations
        # 200 - 10 = 190. 190 / 10 = 19 generations.
        max_generations = (self.budget - pop_size) // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II Loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Tournament selection
                idx1 = self._tournament_select(F, pop_size)
                idx2 = self._tournament_select(F, pop_size)
                
                # Crossover (BLX-alpha)
                p1 = X[idx1]
                p2 = X[idx2]
                # Simple crossover: average + noise
                # Or use BLX-alpha
                alpha = 0.5
                c1 = p1 + alpha * (p2 - p1)
                c2 = p2 + alpha * (p1 - p2)
                # Mutation
                c1 = c1 + 0.1 * (hi - lo) * np.random.standard_normal(self.dim)
                c2 = c2 + 0.1 * (hi - lo) * np.random.standard_normal(self.dim)
                
                # Clip
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                # We only have budget for pop_size offspring per generation
                # So we fill the offspring array
                if i < pop_size // 2:
                    offspring_X[i] = c1
                else:
                    offspring_X[i] = c2
                    # Wait, this logic is flawed. We need exactly pop_size offspring.
                    # Let's just generate one child per loop iteration if pop_size is even?
                    # Better: Generate pop_size children.
            
            # Let's restructure the offspring generation to be cleaner.
            # We need to generate pop_size children.
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                idx1 = self._tournament_select(F, pop_size)
                idx2 = self._tournament_select(F, pop_size)
                p1 = X[idx1]
                p2 = X[idx2]
                
                # BLX-Alpha Crossover
                # x = p1 + alpha * (p2 - p1)
                # This is just a point between them.
                # Let's use a mix of parents and mutation.
                child = p1 + np.random.random() * (p2 - p1)
                # Add mutation
                child = child + 0.2 * (hi - lo) * np.random.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_offspring])
            
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(F_comb)
            
            # Select next generation
            X_next = np.zeros((pop_size, self.dim))
            F_next = np.zeros((pop_size, self.n_obj))
            
            front_idx = 0
            count = 0
            while count < pop_size:
                if front_idx >= len(fronts):
                    break
                front = fronts[front_idx]
                if count + len(front) <= pop_size:
                    # Take whole front
                    for idx in front:
                        X_next[count] = X_comb[idx]
                        F_next[count] = F_comb[idx]
                        count += 1
                    front_idx += 1
                else:
                    # Need to sort this front by crowding distance
                    # Calculate crowding distance for this front
                    cd = self._crowding_distance(F_comb[front])
                    # Sort indices in front by cd descending
                    sorted_indices = np.argsort(-cd)
                    for i in range(pop_size - count):
                        orig_idx = front[sorted_indices[i]]
                        X_next[count] = X_comb[orig_idx]
                        F_next[count] = F_comb[orig_idx]
                        count += 1
                    break
            
            X = X_next
            F = F_next
        
        return X

    def _tournament_select(self, F, k=2):
        # Simple binary tournament
        idxs = np.random.randint(0, len(F), k)
        # Find the non-dominated one among them
        # If one dominates the other, pick it.
        # If not, pick randomly.
        i, j = idxs
        if self._dominates(F[i], F[j]):
            return i
        elif self._dominates(F[j], F[i]):
            return j
        else:
            return np.random.choice([i, j])

    def _dominates(self, a, b):
        # a dominates b if a is better in all objectives and strictly better in at least one
        # Minimizing
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)] # S[i] = set of solutions dominated by i
        n_dom = [0] * n # Number of solutions dominating i
        rank = [0] * n
        
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        for i in range(n):
            if n_dom[i] == 0:
                rank[i] = 0
                fronts[0].append(i)
        
        k = 0
        while len(fronts[k]) > 0:
            k += 1
            fronts.append([])
            for i in fronts[k-1]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k
                        fronts[k].append(j)
                        
        return rank, fronts[:-1] # Exclude the last empty front if any

    def _crowding_distance(self, F):
        # F is the set of solutions in a front
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            obj = F[:, m]
            min_obj = np.min(obj)
            max_obj = np.max(obj)
            if max_obj == min_obj:
                continue
            # Sort by objective m
            sorted_indices = np.argsort(obj)
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            for i in range(1, n - 1):
                cd[sorted_indices[i]] += (obj[sorted_indices[i+1]] - obj[sorted_indices[i-1]]) / (max_obj - min_obj)
        return cd
