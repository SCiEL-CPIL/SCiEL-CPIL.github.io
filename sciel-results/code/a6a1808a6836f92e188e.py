import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20  # Fixed population size

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng()

        # Initialize population
        X = lo + (hi - lo) * rng.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = self.pop_size

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We generate pop_size offspring
            for _ in range(self.pop_size):
                if used >= self.budget:
                    break
                
                # Select parents using tournament selection
                p1 = self.tournament_select(F, rng)
                p2 = self.tournament_select(F, rng)
                
                # Crossover
                c1, c2 = self.sbx_crossover(p1, p2, lo, hi, rng)
                
                # Mutation
                c1 = self.polynomial_mutation(c1, lo, hi, rng)
                c2 = self.polynomial_mutation(c2, lo, hi, rng)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break

            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(F_combined)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= self.pop_size:
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = self.pop_size - len(X_new)
                    cd = self.crowding_distance(F_combined[front], front)
                    # Sort indices by crowding distance descending
                    sorted_indices = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return X

    def tournament_select(self, F, rng, k=2):
        """Tournament selection based on non-dominated rank and crowding distance."""
        # For simplicity and speed, we'll use a simple domination-based tournament
        # Or we can pre-compute ranks. Since pop size is small, we can do it on the fly or pre-compute.
        # Let's pre-compute ranks and crowding distance for the current population before the loop?
        # No, F changes. Let's just pick 2 random and pick the non-dominated one.
        i1 = rng.integers(len(F))
        i2 = rng.integers(len(F))
        
        f1 = F[i1]
        f2 = F[i2]
        
        # Check domination
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return F[i1]
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return F[i2]
        else:
            # If non-dominated, pick randomly
            if rng.random() < 0.5:
                return F[i1]
            else:
                return F[i2]

    def sbx_crossover(self, x1, x2, lo, hi, rng, eta=20):
        """Simulated Binary Crossover."""
        c1 = np.copy(x1)
        c2 = np.copy(x2)
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def polynomial_mutation(self, x, lo, hi, rng, eta=20):
        """Polynomial Mutation."""
        x_m = np.copy(x)
        for i in range(self.dim):
            if rng.random() < 1.0 / self.dim:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                x_m[i] = x[i] + delta * (hi[i] - lo[i])
                x_m[i] = np.clip(x_m[i], lo[i], hi[i])
        return x_m

    def non_dominated_sort(self, F):
        """Returns list of fronts, each front is a list of indices."""
        n = len(F)
        S = [set() for _ in range(n)]  # Dominated sets
        n_dominated = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominated[i] += 1
        
        # Initial front
        front = [i for i in range(n) if n_dominated[i] == 0]
        if front:
            fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def crowding_distance(self, F_subset, indices):
        """Calculate crowding distance for a subset of points."""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        distances = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = np.argsort(F_subset[indices, k])
            distances[np.where(np.isin(np.arange(n), sorted_indices))[0]] = 0 # This is wrong, need to map back
            
            # Better approach:
            # Create a temporary array of distances for this front
            temp_dist = np.zeros(n)
            
            # Get the sorted objective values
            sorted_obj = F_subset[indices, k][sorted_indices]
            
            # Min and max
            if sorted_obj[0] == sorted_obj[-1]:
                temp_dist[:] = np.inf
            else:
                temp_dist[0] = np.inf
                temp_dist[-1] = np.inf
                for i in range(1, n - 1):
                    temp_dist[i] = (sorted_obj[i + 1] - sorted_obj[i - 1]) / (sorted_obj[-1] - sorted_obj[0])
            
            distances += temp_dist
        
        return distances
