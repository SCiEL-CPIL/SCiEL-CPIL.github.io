import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II
        # We need to maintain a population of size pop_size.
        # Each generation:
        # 1. Create offspring of size pop_size.
        # 2. Combine parent and offspring (2 * pop_size).
        # 3. Non-dominated sort.
        # 4. Select pop_size points from the combined set.
        
        # But we have a budget of 200 evaluations.
        # If we do full NSGA-II, we evaluate pop_size offspring per generation.
        # 200 / 20 = 10 generations.
        
        while used + pop_size <= self.budget:
            if time.time() > deadline:
                break
                
            # 1. Create offspring
            X_off = np.zeros((pop_size, self.dim))
            F_off = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Tournament selection
                # Pick 2 random parents, select the non-dominated one.
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # Check dominance
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    parent_idx = p1_idx
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    parent_idx = p2_idx
                else:
                    # Non-dominated, pick random
                    parent_idx = p1_idx if np.random.rand() < 0.5 else p2_idx
                
                parent = X[parent_idx]
                
                # Mutation
                sigma = 0.1
                child = parent + sigma * (hi - lo) * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)
                
                X_off[i] = child
                F_off[i] = problem(child)
                used += 1
                
            # 2. Combine
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])
            
            # 3. Non-dominated sort
            # We need to rank the combined population.
            # Rank 0: Non-dominated
            # Rank 1: Non-dominated after removing Rank 0
            # ...
            
            # We want to select pop_size points.
            # We take all points from Rank 0, then Rank 1, etc.
            # If a rank exceeds the remaining capacity, we use crowding distance.
            
            # For simplicity, let's just take the non-dominated points.
            # If there are more than pop_size non-dominated points, we need to select.
            # If there are fewer, we take the next rank.
            
            # Let's implement a simple non-dominated sort.
            # This is O(N^3) for N=40, which is 64000 ops, very fast.
            
            ranks = np.zeros(2 * pop_size, dtype=int)
            remaining = list(range(2 * pop_size))
            current_rank = 0
            
            while remaining:
                # Find non-dominated in remaining
                nondom_in_remaining = []
                for i in remaining:
                    is_dom = False
                    for j in remaining:
                        if i == j: continue
                        if np.all(F_combined[j] <= F_combined[i]) and np.any(F_combined[j] < F_combined[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        nondom_in_remaining.append(i)
                
                if not nondom_in_remaining:
                    # All dominated, assign next rank
                    for i in remaining:
                        ranks[i] = current_rank
                    break
                
                for i in nondom_in_remaining:
                    ranks[i] = current_rank
                
                remaining = [i for i in remaining if i not in nondom_in_remaining]
                current_rank += 1
            
            # 4. Selection
            # Sort indices by rank
            sorted_indices = np.argsort(ranks)
            
            # Select first pop_size indices
            # But if the last rank is split, we should use crowding distance.
            # For now, let's just take the first pop_size.
            # This might be biased towards the first non-dominated front found.
            
            # Better:
            selected_indices = []
            for rank in range(current_rank):
                rank_indices = [i for i in sorted_indices if ranks[i] == rank]
                if len(selected_indices) + len(rank_indices) <= pop_size:
                    selected_indices.extend(rank_indices)
                else:
                    # Need to select from this rank
                    # Use crowding distance
                    # For simplicity, let's just take random from this rank
                    # or use a simple metric like sum of objectives.
                    # Let's use sum of objectives to prefer "central" points.
                    rank_indices_sorted = sorted(rank_indices, key=lambda i: np.sum(F_combined[i]))
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(rank_indices_sorted[:needed])
                    break
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
            if used >= self.budget:
                break
        
        # If we have remaining budget, use it for single evaluations
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick a random parent
            parent_idx = np.random.randint(pop_size)
            parent = X[parent_idx]
            
            # Mutation
            sigma = 0.05
            child = parent + sigma * (hi - lo) * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            used += 1
            
            # Replace worst
            # Find worst based on sum of objectives
            worst_idx = np.argmax(np.sum(F, axis=1))
            
            if np.sum(f_child) < np.sum(F[worst_idx]):
                X[worst_idx] = child
                F[worst_idx] = f_child
            
        return X
