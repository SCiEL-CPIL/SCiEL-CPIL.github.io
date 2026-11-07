import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        X = []
        F = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        def get_non_dominated(F_list):
            n = len(F_list)
            if n == 0:
                return []
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd[j]:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        nd[i] = False
                        break
                    elif np.all(F_list[i] <= F_list[j]) and np.any(F_list[i] < F_list[j]):
                        nd[j] = False
            return [i for i in range(n) if nd[i]]

        # 1. Initialization: Spread out points
        n_init = 40
        # Use a mix of random and a few structured points
        # Simple random initialization
        for i in range(n_init):
            x = lo + (hi - lo) * np.random.rand(dim)
            eval_point(x)
            
        if not F:
            return X

        F_arr = np.array(F)
        
        # 2. Evolutionary Search with Hypervolume-based Selection
        # We will maintain a population of non-dominated points and use them to generate offspring.
        
        # Reference point for hypervolume calculation (local, for selection only)
        # We can use a dynamic reference point based on the worst observed value + margin
        # Or a fixed one. The task says scoring uses [1705, 12, 0.3].
        # For selection, using the current worst is usually better for relative pressure.
        
        pop_size = 50
        remaining_budget = budget - used
        
        # Adaptive mutation rate
        # Start wide, narrow down
        # Total evaluations ~ 200. 
        # Let's do generations.
        
        # Initialize population with current non-dominated set
        nd_idx = get_non_dominated(F_arr)
        
        # If we have fewer than pop_size non-dominated points, fill with random or best
        if len(nd_idx) < pop_size:
            # Add random points to reach pop_size if possible
            while len(nd_idx) < pop_size and used < budget:
                x = lo + (hi - lo) * np.random.rand(dim)
                f = eval_point(x)
                if f is not None:
                    nd_idx.append(used - 1)
        
        # Ensure we have a valid population
        if len(nd_idx) == 0:
            return X
            
        # Truncate to pop_size
        # We want the "best" points. Since we don't have a single metric, we keep the non-dominated ones.
        # If there are more than pop_size non-dominated, we need to select.
        # Simple selection: keep the ones that contribute most to hypervolume?
        # Or just random subset? Random subset of non-dominated is a valid diversity strategy.
        # Let's keep the first pop_size non-dominated found, or random sample.
        
        # For simplicity and speed, we'll just maintain a list of indices.
        # We will perform a simple evolutionary loop.
        
        # Parameters
        n_offspring = 10
        n_generations = remaining_budget // n_offspring
        
        for gen in range(n_generations):
            if used >= budget or time.time() > deadline:
                break
            
            # Update non-dominated set
            F_arr = np.array(F)
            nd_idx = get_non_dominated(F_arr)
            
            if len(nd_idx) == 0:
                break
            
            # Selection of parents
            # If we have more than 2*pop_size non-dominated, we might want to prune.
            # But for now, let's just pick parents from the current non-dominated set.
            # To ensure diversity, we can pick random parents.
            
            # Generate offspring
            for k in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick two parents
                # Weighted towards "better" points? 
                # Let's just pick uniformly from non-dominated for now to maintain diversity.
                p1_idx = nd_idx[np.random.randint(len(nd_idx))]
                p2_idx = nd_idx[np.random.randint(len(nd_idx))]
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover: Blend
                alpha = np.random.rand()
                x_child = alpha * x1 + (1 - alpha) * x2
                
                # Mutation
                # Adaptive sigma
                # Early: large, Late: small
                progress = used / budget
                sigma = (1.0 - progress) * 0.5 * (hi - lo) + 0.01 * (hi - lo)
                
                x_child = x_child + sigma * np.random.randn(dim)
                x_child = np.clip(x_child, lo, hi)
                
                eval_point(x_child)
                
                # Optional: Local refinement of the new point?
                # No, let's stick to pure evolutionary to save time and complexity.
                # The inspiration said pure evolutionary with HV selection was good.
                # But we are just adding to the global list. The selection for next gen
                # will naturally filter.

        return X
