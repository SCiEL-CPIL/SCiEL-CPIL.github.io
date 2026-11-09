import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        n_gen = self.budget // pop_size
        
        # Initialize population
        pop = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        fitnesses = np.array([problem(x) for x in pop])
        self.eval_count += pop_size
        
        results = []
        for x, f in zip(pop, fitnesses):
            results.append(f)
            
        while self.eval_count < self.budget:
            # Create offspring
            offspring = []
            for _ in range(pop_size):
                # Tournament selection
                i = np.random.randint(pop_size)
                j = np.random.randint(pop_size)
                # Simple dominance check
                if self._dominates(fitnesses[i], fitnesses[j]):
                    p1 = pop[i]
                else:
                    p1 = pop[j]
                    
                i = np.random.randint(pop_size)
                j = np.random.randint(pop_size)
                if self._dominates(fitnesses[i], fitnesses[j]):
                    p2 = pop[i]
                else:
                    p2 = pop[j]
                
                # Simulated Binary Crossover (SBX)
                child = self._sbx(p1, p2, lo, hi)
                
                # Mutation
                child = self._mutate(child, lo, hi)
                
                offspring.append(child)
            
            # Evaluate offspring
            offspring_fitnesses = np.array([problem(x) for x in offspring])
            self.eval_count += pop_size
            
            for x, f in zip(offspring, offspring_fitnesses):
                results.append(f)
                
            # Combine and Select
            combined_pop = np.vstack([pop, np.array(offspring)])
            combined_fit = np.vstack([fitnesses, offspring_fitnesses])
            
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(combined_fit)
            
            # Crowding distance
            crowd = self._crowding_distance(combined_fit, fronts)
            
            # Select next population
            next_pop = []
            next_fit = []
            for front in fronts:
                if len(next_pop) + len(front) <= pop_size:
                    for idx in front:
                        next_pop.append(combined_pop[idx])
                        next_fit.append(combined_fit[idx])
                else:
                    # Sort by crowding distance
                    front_indices = list(front)
                    front_crowds = [crowd[idx] for idx in front_indices]
                    sorted_indices = [front_indices[i] for i in np.argsort(front_crowds)[::-1]]
                    needed = pop_size - len(next_pop)
                    for idx in sorted_indices[:needed]:
                        next_pop.append(combined_pop[idx])
                        next_fit.append(combined_fit[idx])
                    break
            
            pop = np.array(next_pop)
            fitnesses = np.array(next_fit)
            
        return results

    def _dominates(self, f1, f2):
        # f1 dominates f2 if f1 <= f2 for all objectives and f1 < f2 for at least one
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _sbx(self, p1, p2, lo, hi):
        eta = 20
        child = np.zeros_like(p1)
        for i in range(self.dim):
            x1 = p1[i]
            x2 = p2[i]
            if x1 == x2:
                child[i] = x1
                continue
            if np.random.rand() < 0.5:
                beta = 1.0 / (2.0 * np.random.rand() + 1e-9)
            else:
                beta = 1.0 / (2.0 * (1.0 - np.random.rand()) + 1e-9)
            
            if beta > 1.0:
                alpha = 1.0 + (2.0 * (beta - 1.0) * min((x1 - lo[i]) / (hi[i] - lo[i] + 1e-9), (hi[i] - x2) / (hi[i] - lo[i] + 1e-9)))
            else:
                alpha = 1.0 + (2.0 * (1.0 - beta) * max((x1 - lo[i]) / (hi[i] - lo[i] + 1e-9), (hi[i] - x2) / (hi[i] - lo[i] + 1e-9)))
            
            beta = (alpha + 1.0) ** (-1.0 / (eta + 1.0))
            
            x1p = (x1 + x2) / 2.0 + beta * (x1 - x2) / 2.0
            x2p = (x1 + x2) / 2.0 - beta * (x1 - x2) / 2.0
            
            if x1p < lo[i] or x1p > hi[i]:
                child[i] = x2p if lo[i] <= x2p <= hi[i] else x1
            else:
                child[i] = x1p
        return child

    def _mutate(self, child, lo, hi):
        eta = 20
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                x = child[i]
                if np.random.rand() < 0.5:
                    delta = 1.0 - (x - lo[i]) / (hi[i] - lo[i] + 1e-9)
                    beta = (2.0 * delta + 1.0) ** (1.0 / (eta + 1.0))
                else:
                    delta = (hi[i] - x) / (hi[i] - lo[i] + 1e-9)
                    beta = (2.0 * delta + 1.0) ** (1.0 / (eta + 1.0))
                
                x_new = x + (1.0 - beta) * (hi[i] - lo[i]) * (np.random.rand() * 2.0 - 1.0)
                # Simpler mutation: just add noise
                # x_new = x + np.random.normal(0, 0.1 * (hi[i] - lo[i]))
                # Clip
                child[i] = np.clip(x_new, lo[i], hi[i])
        return child

    def _non_dominated_sort(self, fitnesses):
        n = len(fitnesses)
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        current_front = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(fitnesses[i], fitnesses[j]):
                    S[i].add(j)
                elif self._dominates(fitnesses[j], fitnesses[i]):
                    n_dom[i] += 1
        
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        front_idx = 0
        while current_front:
            fronts.append(set(current_front))
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = front_idx + 1
                        next_front.append(j)
            current_front = next_front
            front_idx += 1
            
        return ranks, fronts

    def _crowding_distance(self, fitnesses, fronts):
        n = len(fitnesses)
        crowd = np.zeros(n)
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    crowd[idx] = np.inf
                continue
            
            front_indices = list(front)
            front_fit = fitnesses[front_indices]
            n_front = len(front_indices)
            
            for k in range(self.n_obj):
                sorted_indices = np.argsort(front_fit[:, k])
                crowd[front_indices[sorted_indices[0]]] = np.inf
                crowd[front_indices[sorted_indices[-1]]] = np.inf
                f_range = front_fit[sorted_indices[-1], k] - front_fit[sorted_indices[0], k]
                if f_range == 0:
                    continue
                for i in range(1, n_front - 1):
                    idx = front_indices[sorted_indices[i]]
                    crowd[idx] += (front_fit[sorted_indices[i+1], k] - front_fit[sorted_indices[i-1], k]) / f_range
        return crowd
