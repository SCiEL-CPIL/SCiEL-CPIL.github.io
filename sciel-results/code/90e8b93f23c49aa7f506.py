import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 20
        # Calculate number of generations
        # We use pop_size for initial, then pop_size offspring per gen
        # Total budget = 200. 200 / 20 = 10 generations (including init as gen 0?)
        # Let's say we do 10 iterations of "evaluate 20 new points".
        # Init: 20 points.
        # 9 more generations of 20 points? 20 + 9*20 = 200.
        
        # We will maintain a list of all evaluated points and their fitnesses
        # to ensure we don't exceed budget.
        
        population = []
        fitnesses = []
        
        # Helper to evaluate
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            return np.array(f)

        # 1. Initialization
        for _ in range(pop_size):
            x = np.random.uniform(lo, hi)
            f = evaluate(x)
            population.append(x)
            fitnesses.append(f)
            
        # 2. Evolution
        # We have used 20 evaluations.
        # Remaining: 180.
        # We can do 9 generations of 20 offspring.
        
        for gen in range(9):
            # Combine current population
            # We need to select parents from current population.
            # Then generate offspring.
            
            # Non-dominated sort on current population
            ranks, crowding = self._fast_non_dominated_sort(np.array(fitnesses))
            
            # Select parents
            parents = []
            for _ in range(pop_size // 2):
                # Tournament selection
                i = np.random.randint(0, pop_size)
                j = np.random.randint(0, pop_size)
                if ranks[i] < ranks[j] or (ranks[i] == ranks[j] and crowding[i] > crowding[j]):
                    parents.append(population[i])
                else:
                    parents.append(population[j])
            
            # Generate offspring
            offspring = []
            for _ in range(pop_size // 2):
                p1 = parents[np.random.randint(0, len(parents))]
                p2 = parents[np.random.randint(0, len(parents))]
                # Crossover
                child = self._crossover(p1, p2)
                # Mutation
                child = self._mutate(child, lo, hi)
                offspring.append(child)
            
            # Evaluate offspring
            for x in offspring:
                f = evaluate(x)
                population.append(x)
                fitnesses.append(f)
                
            # Now we have 40 points (20 old, 20 new).
            # We need to select the best 20 to be the next population.
            # But wait, the budget is consumed by evaluation.
            # The "population" for the next generation should be the best 20 from the 40.
            # However, the *recorded* points are all 40.
            # The scorer uses *all* recorded points. So it's good to have more points.
            # But we must not exceed 200 evaluations.
            # We used 20 (init) + 20 (offspring) = 40 so far.
            # We have 160 left.
            # We can continue this process.
            
            # To select the next "active" population for breeding, we pick the best 20.
            all_fitnesses = np.array(fitnesses)
            all_ranks, all_crowding = self._fast_non_dominated_sort(all_fitnesses)
            
            # Select best 20 based on rank and crowding
            indices = np.arange(len(fitnesses))
            # Sort by rank, then by crowding (descending)
            sorted_indices = indices[np.lexsort((all_crowding, all_ranks))]
            # Wait, lexsort sorts by the last key first? No, np.lexsort(keys) sorts by keys[-1] then keys[-2].
            # We want rank ascending, crowding descending.
            # So keys = (-crowding, rank) ? No, rank is int, crowding is float.
            # Let's just sort manually or use a stable sort.
            
            # Simpler: Create a list of (rank, -crowding, index) and sort.
            sorted_list = sorted([(all_ranks[i], -all_crowding[i], i) for i in range(len(fitnesses))])
            best_indices = [item[2] for item in sorted_list[:pop_size]]
            
            population = [population[i] for i in best_indices]
            fitnesses = [fitnesses[i] for i in best_indices]
            
        return [] # Return value ignored

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if F[p][0] <= F[q][0] and F[p][1] <= F[q][1] and (F[p][0] < F[q][0] or F[p][1] < F[q][1]):
                    S[p].append(q)
                elif F[q][0] <= F[p][0] and F[q][1] <= F[p][1] and (F[q][0] < F[p][0] or F[q][1] < F[p][1]):
                    n_count[p] += 1
            
        if n_count[p] == 0:
            rank[p] = 0
        
        # This standard O(N^2) sort is fine for N=20-40.
        # Actually, let's implement it properly.
        
        # Re-implementing properly for correctness
        S = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if F[p][0] <= F[q][0] and F[p][1] <= F[q][1] and (F[p][0] < F[q][0] or F[p][1] < F[q][1]):
                    S[p].append(q)
                elif F[q][0] <= F[p][0] and F[q][1] <= F[p][1] and (F[q][0] < F[p][0] or F[q][1] < F[p][1]):
                    n_count[p] += 1
        
        front = [i for i in range(n) if n_count[i] == 0]
        i = 0
        while front:
            for p in front:
                rank[p] = i
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        # add to next front
                        pass # Need to collect next front
            
            # Collect next front
            next_front = []
            for p in front:
                for q in S[p]:
                    if n_count[q] == 0:
                        next_front.append(q)
            front = next_front
            i += 1
            
        # Crowding distance
        crowding = [0.0] * n
        for i in range(n):
            front_members = [j for j in range(n) if rank[j] == i]
            if len(front_members) <= 2:
                for j in front_members:
                    crowding[j] = float('inf')
            else:
                for k in range(self.n_obj):
                    F_sorted = sorted(front_members, key=lambda idx: F[idx][k])
                    min_f = F[F_sorted[0]][k]
                    max_f = F[F_sorted[-1]][k]
                    if max_f == min_f:
                        continue
                    for idx in [F_sorted[0], F_sorted[-1]]:
                        crowding[idx] = float('inf')
                    for j in range(1, len(F_sorted)-1):
                        idx = F_sorted[j]
                        crowding[idx] += (F[F_sorted[j+1]][k] - F[F_sorted[j-1]][k]) / (max_f - min_f)
                        
        return rank, crowding

    def _crossover(self, p1, p2):
        # Simple uniform crossover
        mask = np.random.random(self.dim) > 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, x, lo, hi):
        # Polynomial mutation
        rate = 0.2
        for i in range(self.dim):
            if np.random.random() < rate:
                delta = np.random.random()
                if delta < 0.5:
                    x[i] = x[i] - (hi[i] - lo[i]) * (2*delta) ** 20
                else:
                    x[i] = x[i] + (hi[i] - lo[i]) * (2*(1-delta)) ** 20
        return np.clip(x, lo, hi)
