import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility if needed, but task says evaluator seeds np.random. 
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)...". 
        # This implies the global numpy random state is seeded. 
        # I should use np.random (global) or a local rng? 
        # "np.random.seed(seed)" is called. So using np.random functions is consistent.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20 # Population size
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Archive: List of non-dominated points
        # We store (f, x) pairs? Or just f? 
        # Hypervolume is calculated on F. 
        # We need to keep the best set of F.
        
        def non_dominated_sort(F):
            # Returns ranks and crowding distances
            n = len(F)
            if n == 0: return [], []
            
            S = [[] for _ in range(n)] # Dominates
            n_d = [0] * n # Number of dominated
            rank = [0] * n
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[j] += 1
            
            front = []
            r = 0
            while True:
                current_front = []
                for i in range(n):
                    if n_d[i] == 0 and rank[i] == -1: # -1 indicates unassigned
                        rank[i] = r
                        current_front.append(i)
                if not current_front: break
                front.append(current_front)
                
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                
                r += 1
            
            # Crowding Distance
            cd = np.zeros(n)
            for f in front:
                if len(f) <= 2:
                    for i in f: cd[i] = np.inf
                    continue
                for k in range(n_obj):
                    vals = F[f, k]
                    min_v = np.min(vals)
                    max_v = np.max(vals)
                    if max_v == min_v:
                        for i in f: cd[i] = np.inf
                        continue
                    # Sort indices by k-th objective
                    sorted_idx = sorted(f, key=lambda i: F[i, k])
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx)-1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i-1]
                        next_idx = sorted_idx[i+1]
                        cd[idx] += (F[next_idx, k] - F[prev_idx, k]) / (max_v - min_v)
                        
            return front, cd

        # Update Archive Logic
        # We want to maintain the Pareto front of ALL evaluated points.
        # With 200 points, we can just keep a list and filter.
        
        archive_F = []
        
        def update_archive(f_vec):
            # Check if f_vec is dominated by any in archive
            # If yes, return
            # If no, remove any in archive dominated by f_vec, add f_vec
            non_dominated = True
            for af in archive_F:
                if np.all(af <= f_vec) and np.any(af < f_vec):
                    non_dominated = False
                    break
            if not non_dominated:
                return
            
            # Remove dominated
            new_archive = []
            for af in archive_F:
                if np.all(f_vec <= af) and np.any(f_vec < af):
                    continue # af is dominated
                new_archive.append(af)
            
            new_archive.append(f_vec)
            archive_F.clear()
            archive_F.extend(new_archive)

        # Initialize Archive
        for f in F:
            update_archive(f)
            
        while evals < budget:
            if time.time() > deadline: break
            
            # Select Parents
            # Simple tournament on rank and crowding
            # For small N, we can just pick 2 random non-dominated points if available
            # Or use the current population's ranks
            
            # Let's use the current population X, F for selection
            # We need to re-sort X, F occasionally? 
            # NSGA-II usually sorts every generation.
            
            # To keep it simple and effective:
            # 1. Sort current population
            # 2. Select 2 parents
            # 3. Crossover/Mutate
            # 4. Evaluate
            # 5. Add to Archive
            # 6. Combine Population + Children, Sort, Select Top N
            
            front, cd = non_dominated_sort(F)
            
            # Create a list of (rank, cd, index)
            # Sort by rank asc, cd desc
            # For tournament, we just need to pick randomly from the top ranks
            
            # Simple approach: Pick 2 random individuals from the current population.
            # Bias towards non-dominated?
            # Let's do a tournament of size 2 based on (rank, cd)
            
            def tournament():
                i = np.random.randint(N)
                j = np.random.randint(N)
                # Get ranks
                # We need a way to get rank of i and j from the sorted fronts
                # This is inefficient to re-sort every step.
                
                # Alternative: Keep the population sorted?
                # Or just use random selection from the non-dominated set if small?
                
                # Let's stick to a simpler heuristic:
                # Pick a random point. If it's dominated by another random point, pick the better one.
                # This is a weak selection.
                
                # Better: Maintain a list of non-dominated indices in the current population.
                # If the list is not empty, pick from there.
                
                # Let's re-evaluate the strategy.
                # With N=20, sorting 20 points is trivial.
                # We can sort every iteration.
                
                # But we need to select parents.
                # Let's just pick 2 random parents from the current population.
                # The diversity is maintained by the archive? No, the archive is separate.
                # The population drives the search.
                
                # Let's use a simple mutation of a random parent.
                # This is basically Random Walk / Simulated Annealing with Archive.
                
                parent_idx = np.random.randint(N)
                parent = X[parent_idx]
                
                # Mutation: DE-like or Gaussian
                # Let's use Gaussian with adaptive step
                # Step size: start 0.5, decay to 0.1
                progress = evals / budget
                step = 0.5 * (1 - progress) + 0.1 * progress
                
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                f_child = problem(child)
                evals += 1
                
                # Update Archive
                update_archive(f_child)
                
                # Update Population
                # Replace the parent? Or replace the worst?
                # Let's replace the parent with the child.
                X[parent_idx] = child
                F[parent_idx] = f_child
                
        return X
