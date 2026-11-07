"""Multi-objective optimizer for RE37 using NSGA-II with DE-based mutation."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                if np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-10:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-10:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def _de_mutation(self, population, CR=0.8, F=0.5):
        """Differential Evolution mutation: DE/rand/1."""
        N = len(population)
        dim = population.shape[1]
        offspring = np.zeros_like(population)
        
        for i in range(N):
            # Select three distinct random indices different from i
            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            
            # DE/rand/1 mutation
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() <= CR or j == j_rand:
                    offspring[i, j] = v[j]
                else:
                    offspring[i, j] = population[i, j]
        
        return offspring

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Data structures
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Halton initialization
        pop_size = 30
        n_init = min(pop_size, n_budget // 2)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span

        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: NSGA-II with DE mutation
        # Maintain a population of size pop_size; use DE to generate offspring
        # Selection via non-dominated sort + crowding distance
        while eval_count < n_budget and time.time() < deadline:
            n_cur = len(X_all)
            if n_cur < 4:
                while n_cur < 4 and eval_count < n_budget:
                    x = lo + span * np.random.random(dim)
                    evaluate(x)
                    n_cur = len(X_all)
                if n_cur < 4:
                    break
            
            X_arr = np.array(X_all)
            F_arr = np.array(F_all)
            
            # Use last min(pop_size, n_cur) points as parent population
            n_parent = min(pop_size, n_cur)
            pop_X = X_arr[-n_parent:]
            
            # Generate offspring using DE
            n_offspring = min(n_parent, n_budget - eval_count)
            if n_offspring <= 0:
                break
            
            offspring = self._de_mutation(pop_X, CR=0.8, F=0.5)
            offspring = np.clip(offspring, lo, hi)
            
            # Evaluate offspring
            for x in offspring:
                if eval_count >= n_budget or time.time() > deadline:
                    break
                evaluate(x)
            
            # Selection: combine parent + offspring, select next generation of size pop_size
            n_total = len(X_all)
            if n_total <= pop_size:
                continue
            
            X_comb = np.array(X_all)
            F_comb = np.array(F_all)
            
            fronts = self._fast_non_dominated_sort(F_comb)
            
            next_gen = []
            for front in fronts:
                if len(next_gen) + len(front) <= pop_size:
                    next_gen.extend(front)
                else:
                    cd = self._crowding_distance(F_comb, front)
                    needed = pop_size - len(next_gen)
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    next_gen.extend(front_sorted[:needed])
                    break
            
            if len(next_gen) == 0:
                next_gen = list(range(min(pop_size, n_total)))
            
            next_gen = sorted(set(next_gen))
            if len(next_gen) >= 2:
                X_all = [X_all[i] for i in next_gen]
                F_all = [F_all[i] for i in next_gen]
        
        return np.array(X_all)
