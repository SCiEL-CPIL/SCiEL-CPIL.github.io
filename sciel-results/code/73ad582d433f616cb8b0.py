"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using adaptive local search on non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 290s to be safe.
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Use np.random to respect the evaluator's seed
        # Initialize with a reasonable population size
        init_size = min(30, self.budget // 5)
        
        # Store all evaluated points to potentially find better non-dominated points later?
        # Actually, just maintaining a list of all points is fine.
        all_X = []
        all_F = []
        used = 0
        
        # Initial Random Sampling
        for _ in range(init_size):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(dim)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x)
            all_F.append(f)
            used += 1
            
        # Helper to get non-dominated indices
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0: return []
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    # j dominates i if j <= i in all and j < i in at least one
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # Adaptive step size logic
        base_step = 0.2 * (hi - lo)
        current_step = base_step
        last_nd_count = len(get_non_dominated_indices(all_F))
        stagnation_counter = 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current non-dominated set
            nd_indices = get_non_dominated_indices(all_F)
            
            if not nd_indices:
                # Fallback to random if something went wrong
                parent_idx = np.random.randint(len(all_X))
            else:
                # Select a parent from the non-dominated set
                # To encourage diversity, we could use crowding distance, 
                # but for small sets, random selection from ND is a good baseline.
                parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            
            parent_x = all_X[parent_idx]
            
            # Generate child via Gaussian mutation
            # Adaptive step: if stagnation, increase step; else decrease
            if len(nd_indices) > last_nd_count:
                # Improved
                current_step = max(base_step * 0.5, current_step * 0.8)
                stagnation_counter = 0
                last_nd_count = len(nd_indices)
            else:
                stagnation_counter += 1
                if stagnation_counter > 10:
                    current_step = min(base_step * 2.0, current_step * 1.2)
                    # If very stuck, consider a random restart (pick random parent instead of ND)
                    if np.random.rand() < 0.2:
                         parent_idx = np.random.randint(len(all_X))
                         parent_x = all_X[parent_idx]

            child_x = parent_x + np.random.randn(dim) * current_step
            child_x = np.clip(child_x, lo, hi)
            
            child_f = np.array(problem(child_x))
            used += 1
            
            all_X.append(child_x)
            all_F.append(child_f)

        return all_X
# EVOLVE-BLOCK-END
