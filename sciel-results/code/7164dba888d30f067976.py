import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        start_time = time.time()
        deadline = start_time + 290
        
        evals_used = 0
        max_evals = self.budget
        
        # NSGA-II Parameters
        pop_size = 15
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            evals_used += 1
            return x_clipped, f

        # 1. Initialization
        rng = np.random.default_rng(0)
        population = []
        for _ in range(pop_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            res = evaluate(x)
            if res is None: break
            population.append(res)
            
        if not population:
            return []

        # Helper: Non-dominated sorting
        def non_dominated_sort(pop):
            n = len(pop)
            S = [set() for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    fi = pop[i][1]
                    fj = pop[j][1]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].add(j)
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        n_count[i] += 1
                        
            for i in range(n):
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
                if next_front:
                    fronts.append(next_front)
                    
            return fronts

        # Helper: Crowding distance
        def crowding_distance(front_indices, pop):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            n = len(front_indices)
            dist = [0.0] * n
            for m in range(self.n_obj):
                sorted_indices = sorted(front_indices, key=lambda i: pop[i][1][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = pop[sorted_indices[0]][1][m]
                f_max = pop[sorted_indices[-1]][1][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (pop[sorted_indices[i+1]][1][m] - pop[sorted_indices[i-1]][1][m]) / f_range
                    
            return dist

        # Helper: Tournament Selection
        def tournament_select(pop, fronts, dists):
            i = rng.integers(len(pop))
            j = rng.integers(len(pop))
            
            rank_i = 0
            for r, front in enumerate(fronts):
                if i in front:
                    rank_i = r
                    break
            
            rank_j = 0
            for r, front in enumerate(fronts):
                if j in front:
                    rank_j = r
                    break
                    
            if rank_i < rank_j:
                return pop[i]
            elif rank_j < rank_i:
                return pop[j]
            else:
                if dists[i] > dists[j]:
                    return pop[i]
                else:
                    return pop[j]

        # Helper: Crossover and Mutation
        def crossover(parent1, parent2):
            x1 = parent1[0]
            x2 = parent2[0]
            u = rng.random(self.dim)
            v = rng.random(self.dim)
            
            def beta(u):
                if u <= 0.5:
                    return (2 * u) ** (1 / 22.0)
                else:
                    return (2 * (1 - u)) ** (1 / 22.0)
                    
            c1 = np.zeros(self.dim)
            c2 = np.zeros(self.dim)
            for i in range(self.dim):
                b1 = beta(u[i])
                b2 = beta(v[i])
                c1[i] = 0.5 * ((1 + b1) * x1[i] + (1 - b1) * x2[i])
                c2[i] = 0.5 * ((1 + b2) * x2[i] + (1 - b2) * x1[i])
                
            def mutate(x):
                sigma = 0.1
                x_new = x + sigma * rng.standard_normal(self.dim)
                return np.clip(x_new, lo, hi)
                
            c1 = mutate(c1)
            c2 = mutate(c2)
            
            return c1, c2

        # Main Loop
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= f) and np.any(archive_F[j] < f):
                    dominated = True
                    break
            if dominated:
                return
            new_archive_X = []
            new_archive_F = []
            for j in range(len(archive_F)):
                if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            archive_X.append(x)
            archive_F.append(f)

        for x, f in population:
            update_archive(x, f)

        gen = 0
        while evals_used < max_evals and time.time() < deadline:
            fronts = non_dominated_sort(population)
            dists = {}
            for front in fronts:
                front_dists = crowding_distance(front, population)
                for idx, d in zip(front, front_dists):
                    dists[idx] = d
            
            offspring = []
            while len(offspring) < pop_size and evals_used + pop_size - len(offspring) <= max_evals:
                p1 = tournament_select(population, fronts, dists)
                p2 = tournament_select(population, fronts, dists)
                c1, c2 = crossover(p1, p2)
                
                res1 = evaluate(c1)
                if res1:
                    offspring.append(res1)
                    update_archive(res1[0], res1[1])
                    
                if evals_used >= max_evals:
                    break
                    
                res2 = evaluate(c2)
                if res2:
                    offspring.append(res2)
                    update_archive(res2[0], res2[1])
                    
            if not offspring:
                break
                
            combined = population + offspring
            fronts = non_dominated_sort(combined)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend([combined[i] for i in front])
                else:
                    dists = {}
                    front_dists = crowding_distance(front, combined)
                    for idx, d in zip(front, front_dists):
                        dists[idx] = d
                    sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                    needed = pop_size - len(new_population)
                    new_population.extend([combined[i] for i in sorted_front[:needed]])
                    break
                    
            population = new_population
            gen += 1

        return archive_X
