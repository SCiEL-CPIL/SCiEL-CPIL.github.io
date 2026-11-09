import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Heuristic population size: small enough to fit in budget, large enough for diversity
        self.pop_size = 20

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in tests if needed, but task says evaluator seeds np.random
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # NSGA-II parameters
        # Crossover probability and distribution index
        pc = 0.9
        eta_c = 20.0
        # Mutation probability and distribution index
        pm = 1.0 / self.dim
        eta_m = 20.0
        
        while used + self.pop_size <= self.budget:
            # Create offspring
            offspring_X = []
            for _ in range(self.pop_size):
                # Tournament selection for parent 1
                p1 = self._tournament_select(F, rng)
                # Tournament selection for parent 2
                p2 = self._tournament_select(F, rng)
                
                # Crossover
                child = self._sbx_crossover(p1, p2, lo, hi, rng, pc, eta_c)
                
                # Mutation
                child = self._polynomial_mutation(child, lo, hi, rng, pm, eta_m)
                
                offspring_X.append(child)
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += self.pop_size
            
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Fast non-dominated sort
            ranks, fronts = self._fast_non_dominated_sort(F_all)
            
            # Select next population
            next_X = []
            next_F = []
            current_front_idx = 0
            while len(next_X) < self.pop_size:
                if current_front_idx >= len(fronts):
                    break
                front = fronts[current_front_idx]
                
                # If this front fits entirely, add it
                if len(next_X) + len(front) <= self.pop_size:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    current_front_idx += 1
                else:
                    # Need to select from this front based on crowding distance
                    front_F = F_all[front]
                    distances = self._crowding_distance(front_F)
                    # Sort indices in front by distance (descending)
                    sorted_indices = np.argsort(-distances)
                    needed = self.pop_size - len(next_X)
                    for i in range(needed):
                        idx = front[sorted_indices[i]]
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break
                    
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X

    def _tournament_select(self, F, rng, k=2):
        # Non-dominated sorting for tournament
        ranks, _ = self._fast_non_dominated_sort(F)
        # Calculate crowding distance for all points for a fair tournament? 
        # Standard NSGA-II tournament uses rank then crowding. 
        # To be efficient, we can just pick k random indices and compare.
        idxs = rng.integers(0, len(F), k)
        best_idx = idxs[0]
        for i in range(1, k):
            if ranks[idxs[i]] < ranks[best_idx]:
                best_idx = idxs[i]
            elif ranks[idxs[i]] == ranks[best_idx]:
                # If same rank, we need crowding distance. 
                # This is expensive to calculate for just a tournament.
                # Approximation: just pick randomly if same rank, or compute distance for these k.
                # Let's compute distance for the candidates involved.
                # For simplicity and speed in this small budget, let's just pick randomly if ranks are equal.
                # This might hurt diversity slightly but keeps code simple.
                # Alternatively, we can store crowding distances from the previous generation.
                # Let's stick to simple rank-based tournament for now, or implement a proper one.
                # Proper NSGA-II requires crowding distance in tournament.
                # Let's compute crowding distance for the whole population once per generation? No, that's what we do for selection.
                # For tournament, we can just use rank. If ranks are equal, pick randomly.
                if rng.random() < 0.5:
                    best_idx = idxs[i]
        return X[best_idx] # Wait, X is not passed. Need to pass X or return index.
        
    # Correction: _tournament_select needs X to return the vector.
    # Let's refactor to return index or vector.
    
    def _tournament_select(self, F, X, rng, k=2):
        ranks, _ = self._fast_non_dominated_sort(F)
        idxs = rng.integers(0, len(F), k)
        best_idx = idxs[0]
        # We need crowding distance for tie-breaking. 
        # Calculating it for the whole population every tournament is too slow.
        # We will calculate it once per generation and pass it? 
        # Or just use rank. Given the small budget, rank-only tournament might be sufficient.
        for i in range(1, k):
            if ranks[idxs[i]] < ranks[best_idx]:
                best_idx = idxs[i]
            elif ranks[idxs[i]] == ranks[best_idx]:
                # Tie-break randomly
                if rng.random() < 0.5:
                    best_idx = idxs[i]
        return X[best_idx]

    def _sbx_crossover(self, p1, p2, lo, hi, rng, pc, eta):
        if rng.random() > pc:
            return p1.copy()
        
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                # Crossover this gene
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = 1.0 + (2.0 * min(p1[i], p2[i]) - lo[i]) / (p1[i] - p2[i])
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        c1 = 0.5 * (p1[i] + p2[i] - (p2[i] - p1[i]) * (alpha - 1.0))
                    else:
                        beta = 1.0 + (2.0 * (hi[i] - max(p1[i], p2[i]))) / (p1[i] - p2[i])
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        c1 = 0.5 * (p1[i] + p2[i] + (p2[i] - p1[i]) * (alpha - 1.0))
                    child[i] = c1
                else:
                    child[i] = p1[i]
            else:
                child[i] = p1[i]
        return np.clip(child, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, rng, pm, eta):
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < pm:
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                rnd = rng.random()
                mut_pow = 1.0 / (eta + 1.0)
                if rnd < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                x_mut[i] += deltaq * (hi[i] - lo[i])
        return np.clip(x_mut, lo, hi)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)] # Dominated sets
        n_dom = np.zeros(n, dtype=int) # Number of points dominating
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        rank = 0
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                ranks[i] = rank
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1
            
        # Handle any remaining points (should not happen if logic is correct)
        for i in range(n):
            if ranks[i] == 0 and i not in [idx for front in fronts for idx in front]:
                 # This case is tricky, usually all points are assigned
                 pass

        return ranks, fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        for m in range(F.shape[1]):
            # Sort by objective m
            order = np.argsort(F[:, m])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            f_min = F[order[0], m]
            f_max = F[order[-1], m]
            if f_max == f_min:
                continue
            for i in range(1, n - 1):
                distances[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (f_max - f_min)
        
        return distances
