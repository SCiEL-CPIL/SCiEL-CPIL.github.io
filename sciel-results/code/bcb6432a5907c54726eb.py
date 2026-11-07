import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # With 200 budget, P=20 allows 10 generations.
        # P=10 allows 20 generations.
        # Let's try P=20 for better diversity in DE.
        pop_size = 20
        
        rng = np.random.default_rng(0)
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring using DE
            V = np.empty_like(X)
            for i in range(pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Avoid picking i itself? DE usually allows it but distinct is better
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                F_factor = 0.8 # Mutation factor
                V[i] = X[r1] + F_factor * (X[r2] - X[r3])
                V[i] = np.clip(V[i], lo, hi)
            
            # Evaluate offspring
            F_v = np.array([problem(x) for x in V])
            used += pop_size
            
            # Combine
            X_all = np.vstack([X, V])
            F_all = np.vstack([F, F_v])
            
            # Non-dominated Sorting
            # Find non-dominated points in X_all
            # Dominance: A dominates B if A <= B and A != B
            # Vectorized check:
            # diff = F_all[:, np.newaxis, :] - F_all[np.newaxis, :, :]
            # less_eq = np.all(diff <= 0, axis=2)
            # not_equal = np.any(diff != 0, axis=2)
            # dominates = less_eq & not_equal
            # nondom_mask = ~np.any(dominates, axis=1) # i is non-dominated if no j dominates i
            
            # This is O(N^2 * n_obj). N=40, n_obj=3. 40*40*3 = 4800 ops. Very fast.
            
            diff = F_all[:, np.newaxis, :] - F_all[np.newaxis, :, :]
            less_eq = np.all(diff <= 0, axis=2)
            not_equal = np.any(diff != 0, axis=2)
            dominates = less_eq & not_equal
            
            # A point i is non-dominated if there is no j such that j dominates i
            # dominates[j, i] is True if j dominates i
            # So i is dominated if np.any(dominates[:, i])
            is_dominated = np.any(dominates, axis=0)
            nondom_indices = np.where(~is_dominated)[0]
            
            # If we have more than pop_size non-dominated points, we need to truncate
            if len(nondom_indices) > pop_size:
                # Use simple HV contribution or just random?
                # Better: Use crowding distance or simple penalty based on objective sum
                # For speed, let's just pick the ones with smallest sum of objectives? 
                # No, that biases towards one corner.
                # Let's use a simple approximation: sort by sum of normalized objectives?
                # Or just keep the first pop_size? 
                # To maintain diversity, we can use a greedy HV maximization on the non-dominated set.
                # But with N=40, we can afford a simple greedy selection.
                
                # Simple approach: Randomly select? No.
                # Let's use a simplified crowding distance or just keep the ones that are "extreme".
                # Actually, for WFG, the front is smooth. 
                # Let's just keep the non-dominated points. If > pop_size, we need to reduce.
                # We can assign a rank. Rank 1 is non-dominated.
                # If we have too many, we can just pick the ones with the lowest "regret" or similar.
                # Given the small size, let's just pick the ones that maximize the minimum distance to other points in the non-dominated set?
                
                # Simpler: Just take the first pop_size non-dominated points? 
                # This might bias.
                # Let's try a simple HV-based selection.
                # Reference point: [3, 5, 7]
                ref = np.array([3.0, 5.0, 7.0])
                
                # Calculate contribution of each non-dominated point to HV?
                # That's expensive.
                # Let's just use a random subset of the non-dominated points if there are too many?
                # Or better, sort by the sum of objectives?
                
                # Let's stick to a simple heuristic: keep the points that are "most spread out".
                # Or just keep the non-dominated points and let the population size grow?
                # No, we need fixed pop size for DE.
                
                # Let's use a simple greedy HV maximization.
                selected = []
                remaining = list(nondom_indices)
                
                # To save time, if len(remaining) is small, do it.
                # We need to pick pop_size points.
                # Greedy HV:
                # 1. Start with empty set.
                # 2. Add point that maximizes HV increment.
                # This is O(K * N * n_obj) where K is target size.
                # K=20, N=40. 20*40*3 = 2400. Fast.
                
                def hv_increment(point, current_set, ref):
                    # Calculate HV of current_set U {point} - HV of current_set
                    # Approximation: Volume of box defined by point and ref, minus overlap?
                    # Too complex.
                    # Let's just use a simple proxy: minimize the sum of distances to ref?
                    # Or just pick random?
                    
                    # Let's try: Sort by sum of objectives and pick spread?
                    pass
                
                # Actually, for this budget, a simpler truncation is fine.
                # Let's just pick the non-dominated points. If there are more than pop_size,
                # we can just take the ones with the lowest "dominance count" (which is 0 for all)
                # and then break ties by random.
                # Or better: Use the first pop_size found?
                
                # Let's use a simple diversity measure:
                # Sort by the maximum distance to any other point in the non-dominated set?
                
                # For now, let's just take a random subset of the non-dominated points
                # to ensure diversity is maintained stochastically.
                if len(nondom_indices) > pop_size:
                    selected_idx = rng.choice(nondom_indices, size=pop_size, replace=False)
                else:
                    selected_idx = nondom_indices
                
                # If we have fewer than pop_size non-dominated points, we need to fill the rest
                # with the best dominated points.
                if len(selected_idx) < pop_size:
                    dominated_indices = np.where(is_dominated)[0]
                    # Sort dominated by sum of objectives?
                    sums = np.sum(F_all[dominated_indices], axis=1)
                    best_dominated = dominated_indices[np.argsort(sums)[:pop_size - len(selected_idx)]]
                    selected_idx = np.concatenate([selected_idx, best_dominated])
                
                X = X_all[selected_idx]
                F = F_all[selected_idx]
            else:
                # If exactly pop_size or less non-dominated, keep them all?
                # If less, we need to fill.
                if len(nondom_indices) < pop_size:
                    dominated_indices = np.where(is_dominated)[0]
                    sums = np.sum(F_all[dominated_indices], axis=1)
                    best_dominated = dominated_indices[np.argsort(sums)[:pop_size - len(nondom_indices)]]
                    final_idx = np.concatenate([nondom_indices, best_dominated])
                else:
                    final_idx = nondom_indices
                
                X = X_all[final_idx]
                F = F_all[final_idx]
        
        return X
