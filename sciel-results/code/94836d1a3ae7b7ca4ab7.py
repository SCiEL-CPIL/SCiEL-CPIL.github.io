import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Strategy: Use a mix of global exploration (DE) and local exploitation
        # Given the small budget (200), we need to be efficient.
        # WFG7 is known to be difficult and multimodal.
        # We will use a population-based approach with DE operators, similar to the inspiration that scored 1.07.
        
        # Parameters
        pop_size = 20
        # We have 200 evaluations. 
        # Init: 20
        # Generations: (200 - 20) / 20 = 9 generations of 20 offspring each.
        
        # Initialize Population
        # Use a stratified sampling or just random? Random is fine with DE.
        # Let's try a slightly smarter initialization: mix of random and a few structured points?
        # For simplicity and robustness, random uniform.
        X = np.random.rand(pop_size, dim) * (hi - lo) + lo
        F = np.array([problem(x) for x in X])
        
        used = pop_size
        
        # Track best hypervolume or just maintain the population?
        # The evaluator records ALL returned objective vectors.
        # So we just need to generate a good set of points.
        # The final return value is ignored, but the *calls* to problem() are what matter.
        # So we just need to make 200 calls that yield a diverse and good Pareto front.
        
        # DE Parameters
        # F (scale) and CR (crossover rate)
        F_de = 0.8
        CR_de = 0.9
        
        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate Offspring
            # We will generate pop_size offspring at a time to match the population size for selection.
            # If remaining budget < pop_size, we just fill up what we can.
            batch_size = min(pop_size, budget - used)
            
            offspring = np.zeros((batch_size, dim))
            
            for i in range(batch_size):
                # DE/rand/1
                # Select 3 distinct parents
                idxs = np.random.choice(pop_size, size=3, replace=False)
                p1, p2, p3 = X[idxs[0]], X[idxs[1]], X[idxs[2]]
                base = p1
                donor = p2 + F_de * (p3 - p2)
                
                # Crossover
                j_rand = np.random.randint(0, dim)
                bin_mask = np.random.rand(dim) <= CR_de
                bin_mask[j_rand] = True # Ensure at least one gene from donor
                
                trial = np.where(bin_mask, donor, base)
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                offspring[i] = trial
            
            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += batch_size
            
            # Combine Parents and Offspring
            X_combined = np.vstack((X, offspring))
            F_combined = np.vstack((F, F_offspring))
            
            # Selection: Non-Dominated Sorting + Crowding Distance
            # We need to select the best 'pop_size' from the combined 2*pop_size (or less if batch was small)
            # Actually, if batch_size < pop_size, the combined size is pop_size + batch_size.
            # We should select the best pop_size to keep the population size constant for the next iteration.
            
            fronts = self._fast_non_dominated_sort(F_combined)
            
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            current_front_idx = 0
            selected_count = 0
            
            while selected_count < pop_size and current_front_idx < len(fronts):
                front_indices = fronts[current_front_idx]
                
                if selected_count + len(front_indices) <= pop_size:
                    # Add entire front
                    for idx in front_indices:
                        new_X[selected_count] = X_combined[idx]
                        new_F[selected_count] = F_combined[idx]
                        selected_count += 1
                    current_front_idx += 1
                else:
                    # Need to select from this front based on crowding distance
                    front_F = F_combined[front_indices]
                    crowding = self._crowding_distance(front_F)
                    
                    # Sort by crowding distance in descending order
                    sorted_indices = np.argsort(-crowding)
                    
                    needed = pop_size - selected_count
                    for j in range(needed):
                        idx = front_indices[sorted_indices[j]]
                        new_X[selected_count] = X_combined[idx]
                        new_F[selected_count] = F_combined[idx]
                        selected_count += 1
                    break
            
            X = new_X
            F = new_F

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
            
        S = [set() for _ in range(n)]  # Dominated sets
        n_dom = [0] * n  # Number of solutions dominating i
        fronts = [set() for _ in range(n)]  # Fronts
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                # i dominates j if F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        # First front
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].add(i)
        
        k = 1
        while len(fronts[k - 1]) > 0 and k < n:
            next_front = set()
            for i in fronts[k - 1]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            fronts[k] = next_front
            k += 1
        
        # Convert to lists for easier indexing
        return [list(front) for front in fronts if len(front) > 0]

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort indices by objective k
            idx_sorted = np.argsort(F[:, k])
            dist[idx_sorted[0]] = np.inf
            dist[idx_sorted[-1]] = np.inf
            
            f_range = F[idx_sorted[-1], k] - F[idx_sorted[0], k]
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                dist[idx_sorted[i]] += (F[idx_sorted[i + 1], k] - F[idx_sorted[i - 1], k]) / f_range
        
        return dist
