"""Multi-objective optimizer using Differential Evolution with Non-Dominated Sorting (NSGA-II style)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for scoring (task specific)
        # Note: The scoring uses [3000.0, 0.06]. We use a dynamic reference for sorting 
        # to ensure we cover the front, but we must be careful not to waste evaluations 
        # on points that are terrible. However, for DE, we just need to evolve.
        
        # Population size: Keep it moderate. Budget is 200.
        # If pop=10, we have 20 generations.
        # If pop=20, we have 10 generations.
        # DE works well with pop >= 5 * dim. dim=4, so pop=20 is good.
        pop_size = 20
        if self.budget < pop_size * 2:
            pop_size = max(5, self.budget // 2)
            
        # Initialize population uniformly
        # Use Latin Hypercube for better initial spread
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            segments = np.linspace(lo[i], hi[i], pop_size + 1)
            offsets = np.random.random(pop_size) * (segments[1] - segments[0])
            X[:, i] = segments[:-1] + offsets
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper for non-dominated sorting
        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts, where each front is a list of indices.
            F is a numpy array of shape (N, n_obj).
            """
            N = F.shape[0]
            domination_count = np.zeros(N, dtype=int)
            dominated_set = [[] for _ in range(N)]
            fronts = []
            
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
                
            # First front
            current_front = [i for i in range(N) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)
                
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                k += 1
                if k > N: break # Safety break
            
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a front.
            Returns a dictionary {index: distance}
            """
            if not front_indices:
                return {}
            
            m = len(front_indices)
            if m <= 2:
                return {idx: float('inf') for idx in front_indices}
            
            distances = {idx: 0.0 for idx in front_indices}
            
            for k in range(self.n_obj):
                # Sort indices by objective k
                sorted_indices = sorted(front_indices, key=lambda idx: F[idx, k])
                
                # Boundary points get infinite distance
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                # Normalize objective range
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                
                for i in range(1, m - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    distances[idx] += (F[next_idx, k] - F[prev_idx, k]) / f_range
                    
            return distances

        # Main Evolution Loop
        # We maintain a population. Each generation, we generate offspring.
        # Selection: Tournament selection based on non-dominated sorting and crowding.
        
        stagnation_count = 0
        best_hv = -np.inf
        ref_point = np.array([3000.0, 0.06]) # For HV calculation to track progress
        
        def calc_hv(F_set):
            """Calculate HV for a set of points relative to ref_point. Only 2D case."""
            if not F_set:
                return 0.0
            pts = [f for f in F_set if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts.sort(key=lambda x: x[0])
            hv = 0.0
            for i in range(len(pts)):
                f1 = pts[i][0]
                f2 = pts[i][1]
                next_f1 = pts[i+1][0] if i + 1 < len(pts) else ref_point[0]
                w1 = next_f1 - f1
                if w1 <= 0: continue
                h2 = ref_point[1] - f2
                if h2 <= 0: continue
                hv += w1 * h2
            return hv

        # Initial HV
        current_hv = calc_hv(F)
        best_hv = current_hv

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            n_offspring = pop_size
            X_offspring = np.zeros((n_offspring, self.dim))
            
            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                    
                # Differential Evolution: x_new = x_rand1 + F * (x_rand2 - x_rand3)
                # Pick 3 distinct parents
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation factor F (usually 0.5 to 1.0)
                F_mut = 0.8
                # Crossover rate CR (usually 0.5 to 1.0)
                CR = 0.5
                
                # Vector difference
                diff = X[r2] - X[r3]
                
                # Trial vector
                v = X[r1] + F_mut * diff
                
                # Binomial Crossover
                x_trial = X[r1].copy()
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        x_trial[j] = v[j]
                        
                x_trial = np.clip(x_trial, lo, hi)
                
                # Evaluate
                f_trial = problem(x_trial)
                used += 1
                
                X_offspring[i] = x_trial
                # Store f_trial temporarily, will evaluate all at end or keep in list
                # To avoid storing F_offspring separately, let's just evaluate and store in a list
                # Actually, let's collect all offspring and evaluate them in a batch to be clean?
                # No, problem(x) is a function call. We must call it.
                # Let's store F_offspring
                if not hasattr(self, '_temp_F_off'):
                    self._temp_F_off = []
                self._temp_F_off.append(f_trial)
                
            # If we broke early, resize
            if used < self.budget and len(self._temp_F_off) < n_offspring:
                 # This shouldn't happen unless deadline hit, but handle gracefully
                 pass
            
            F_offspring = np.array(self._temp_F_off)
            del self._temp_F_off
            
            # Combine Parent and Offspring
            X_all = np.vstack([X, X_offspring])
            F_all = np.vstack([F, F_offspring])
            
            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(F_all)
            
            # Select next population
            X_new = np.zeros((pop_size, self.dim))
            F_new = np.zeros((pop_size, self.n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_new[count] = X_all[idx]
                        F_new[count] = F_all[idx]
                        count += 1
                else:
                    # Take only some, need crowding distance
                    # Calculate crowding for this front
                    # Note: F_all is the full array, front indices are global
                    # We need crowding distance for the individuals in this front
                    # The standard algorithm calculates it on the front itself.
                    
                    # Extract F for this front to calculate distances
                    F_front = F_all[front]
                    
                    # Map global indices to local indices for sorting?
                    # Crowding distance depends on the values, not the indices.
                    # But we need to assign the distance back to the global index.
                    
                    # Let's implement a simpler crowding distance that works on indices
                    # and the global F array.
                    
                    m = len(front)
                    if m <= 2:
                        # Take all
                        for idx in front:
                            X_new[count] = X_all[idx]
                            F_new[count] = F_all[idx]
                            count += 1
                            if count == pop_size: break
                        break
                    
                    # Calculate distances
                    dists = np.zeros(m)
                    for k in range(self.n_obj):
                        # Sort front indices by objective k
                        sorted_local = np.argsort(F_front[:, k])
                        # Get global indices in this order
                        sorted_global = [front[i] for i in sorted_local]
                        
                        # Assign inf to boundaries
                        dists[sorted_local[0]] = float('inf')
                        dists[sorted_local[-1]] = float('inf')
                        
                        f_min = F_front[sorted_local[0], k]
                        f_max = F_front[sorted_local[-1], k]
                        f_range = f_max - f_min
                        if f_range == 0:
                            continue
                        
                        for i in range(1, m - 1):
                            local_idx = sorted_local[i]
                            prev_local = sorted_local[i - 1]
                            next_local = sorted_local[i + 1]
                            dists[local_idx] += (F_front[next_local, k] - F_front[prev_local, k]) / f_range
                    
                    # Sort front by distance descending
                    # We want to pick the ones with highest distance
                    # Create list of (distance, global_idx)
                    candidates = [(dists[i], front[i]) for i in range(m)]
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    needed = pop_size - count
                    for i in range(needed):
                        _, global_idx = candidates[i]
                        X_new[count] = X_all[global_idx]
                        F_new[count] = F_all[global_idx]
                        count += 1
                    break
                    
            # Update population
            X = X_new
            F = F_new
            
            # Check stagnation
            new_hv = calc_hv(F)
            if new_hv > best_hv * 1.001: # 0.1% improvement
                best_hv = new_hv
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # If stagnating, inject random diversity
            if stagnation_count > 3:
                # Replace 20% of population with random points
                n_random = max(1, pop_size // 5)
                idxs = np.random.choice(pop_size, n_random, replace=False)
                X_random = lo + (hi - lo) * np.random.random((n_random, self.dim))
                F_random = np.array([problem(x) for x in X_random])
                used += n_random
                
                for i in range(n_random):
                    X[idxs[i]] = X_random[i]
                    F[idxs[i]] = F_random[i]
                    
                stagnation_count = 0
                # Recalculate HV if needed? Not strictly necessary for loop logic, 
                # but good for consistency.
                # best_hv = calc_hv(F) # Optional

        return X # Return value is ignored

# EVOLVE-BLOCK-END
