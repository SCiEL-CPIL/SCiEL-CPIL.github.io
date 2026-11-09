import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # Check if p dominates q
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    n_dom[p] += 1
                # Check if q dominates p
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dom[q] += 1

        # First front
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts[:-1] if len(fronts) > 1 else fronts

    def _crowding_distance(self, F, front_indices):
        """
        Calculate crowding distance for a given front.
        """
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n

        dist = [0.0] * n
        m = self.n_obj

        for k in range(m):
            # Sort indices by k-th objective
            sorted_indices = sorted(front_indices, key=lambda idx: F[idx][k])
            
            # Assign infinite distance to boundary points
            dist[sorted_indices[0]] = float('inf')
            dist[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0]][k]
            f_max = F[sorted_indices[-1]][k]
            range_val = f_max - f_min
            if range_val == 0:
                continue

            for i in range(1, n - 1):
                dist[sorted_indices[i]] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / range_val

        return dist

    def _tournament_select(self, ranks, crowding, size=2):
        """
        Tournament selection based on rank and crowding distance.
        """
        indices = np.random.randint(0, len(ranks), size)
        best_idx = indices[0]
        for i in indices[1:]:
            if ranks[i] < ranks[best_idx]:
                best_idx = i
            elif ranks[i] == ranks[best_idx]:
                if crowding[i] > crowding[best_idx]:
                    best_idx = i
        return best_idx

    def _sbx_crossover(self, x1, x2, bounds, eta_c=20.0):
        """
        Simulated Binary Crossover.
        """
        x3 = np.zeros_like(x1)
        x4 = np.zeros_like(x2)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x2[i] - x1[i]) > 1e-14:
                    if np.random.rand() <= 1.0 / (eta_c + 1.0):
                        beta = (2.0 * np.random.rand()) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta_c + 1.0))
                    
                    x3[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    x4[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    
                    # Clip to bounds
                    x3[i] = np.clip(x3[i], lo[i], hi[i])
                    x4[i] = np.clip(x4[i], lo[i], hi[i])
                else:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
            else:
                x3[i] = x1[i]
                x4[i] = x2[i]
        
        return x3, x4

    def _poly_mutation(self, x, bounds, eta_m=20.0):
        """
        Polynomial Mutation.
        """
        x_mut = x.copy()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Mutation probability 1/dim is common, but 0.5 per gene is aggressive. Let's use 1/dim.
                # Actually, standard NSGA-II uses p_m = 1/dim.
                pass 
            # Let's stick to 1/dim for standard behavior
            if np.random.rand() < 1.0 / self.dim:
                if hi[i] - lo[i] < 1e-14:
                    continue
                y = x_mut[i]
                if np.random.rand() <= 0.5:
                    u = np.random.rand()
                    delta1 = (y - lo[i]) / (hi[i] - lo[i])
                    xq = 1.0 + (1.0 - delta1)
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xq ** (eta_m + 1.0))
                    deltaq = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    y = y + deltaq * (hi[i] - lo[i])
                else:
                    u = np.random.rand()
                    delta2 = (hi[i] - y) / (hi[i] - lo[i])
                    xq = 1.0 + delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 1.0) * (xq ** (eta_m + 1.0))
                    deltaq = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    y = y + deltaq * (hi[i] - lo[i])
                
                x_mut[i] = np.clip(y, lo[i], hi[i])
        
        return x_mut

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200, we can afford a smaller population.
        # Let's try N=20. Generations = 200/20 = 10.
        # Or N=10. Generations = 20.
        # Let's use N=20 for better diversity in early stages.
        N = 20
        
        # Initial Population
        # Use Sobol for better spread than random
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X = sobol.random(N)
        X = lo + X * (hi - lo)
        
        all_X = []
        all_F = []
        
        # Evaluate initial population
        for x in X:
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
        
        X = np.array(all_X)
        F = np.array(all_F)
        
        # Main Evolutionary Loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Check budget
            if len(all_X) + N > self.budget:
                # If we can't fit a full generation, stop or do partial?
                # Stopping is safer to avoid breaking flow.
                break
                
            # 1. Non-dominated sorting and crowding distance
            fronts = self._fast_non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            crowding = np.zeros(N)
            
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
                dists = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    crowding[idx] = dists[j]
            
            # 2. Generate Offspring
            offspring_X = []
            offspring_F = []
            
            # We need to generate N offspring.
            # But we must respect the budget.
            # We will generate pairs.
            for _ in range(N // 2):
                if len(all_X) + len(offspring_X) + 2 > self.budget:
                    break
                    
                # Select parents
                idx1 = self._tournament_select(ranks, crowding)
                idx2 = self._tournament_select(ranks, crowding)
                
                # Crossover
                c1, c2 = self._sbx_crossover(X[idx1], X[idx2], bounds)
                
                # Mutation
                c1 = self._poly_mutation(c1, bounds)
                c2 = self._poly_mutation(c2, bounds)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                # Record for global archive
                all_X.append(c1)
                all_F.append(f1)
                all_X.append(c2)
                all_F.append(f2)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # 3. Combine Parent and Offspring
            X_combined = np.vstack((X, offspring_X))
            F_combined = np.vstack((F, offspring_F))
            
            # 4. Selection (Elitism)
            fronts = self._fast_non_dominated_sort(F_combined)
            
            new_X = []
            new_F = []
            n_selected = 0
            for front in fronts:
                if n_selected + len(front) <= N:
                    # Take all
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_selected += len(front)
                else:
                    # Need to select from this front based on crowding
                    needed = N - n_selected
                    dists = self._crowding_distance(F_combined, front)
                    
                    # Sort by crowding distance descending
                    sorted_pairs = sorted(zip(front, dists), key=lambda x: x[1], reverse=True)
                    for idx, _ in sorted_pairs[:needed]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_selected += needed
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            if time.time() > deadline - 10.0:
                break

        return np.array(all_X)
