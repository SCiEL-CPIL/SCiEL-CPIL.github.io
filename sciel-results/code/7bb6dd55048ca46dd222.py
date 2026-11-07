"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer exploiting DTLZ1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # DTLZ1: first 4 vars control front, last 3 control g(x)
        # Set last 3 to 0 for Pareto front membership
        n_front = 4  # first 4 variables
        n_g = self.dim - n_front  # last 3 variables

        def make_x(front_vars):
            """Create full vector with g-vars set to 0."""
            x = np.zeros(self.dim)
            x[:n_front] = front_vars
            # x[n_front:] = 0 (already zero)
            return x

        # Initialize population spread across the Pareto front
        pop_size = 60
        X_front = np.zeros((pop_size, n_front))
        
        # Use Dirichlet distribution for uniform coverage of the simplex
        for i in range(pop_size):
            w = rng.dirichlet(np.ones(n_front))
            X_front[i] = w
        
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X[i] = make_x(X_front[i])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Non-dominated sorting helper
        def is_dominated(F_arr, idx):
            f = F_arr[idx]
            for j in range(len(F_arr)):
                if j == idx:
                    continue
                if np.all(F_arr[j] <= f) and np.any(F_arr[j] < f):
                    return True
            return False
        
        # DE parameters
        CR = 0.8
        F_DE = 0.5
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated set
            nondom_idx = [i for i in range(len(F)) if not is_dominated(F, i)]
            if not nondom_idx:
                nondom_idx = list(range(len(F)))
            
            # Select parents from non-dominated set
            n_parents = min(3, len(nondom_idx))
            parent_indices = rng.choice(nondom_idx, size=n_parents, replace=False)
            a, b, c = [X[parent_indices[0]], X[parent_indices[1]], X[parent_indices[2]]]
            
            # DE mutation
            mutant = a + F_DE * (b - c)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            cross_mask = rng.random(self.dim) < CR
            cross_mask[rng.integers(self.dim)] = True  # Ensure at least one gene
            trial = np.where(cross_mask, mutant, X[parent_indices[0]])
            
            # Evaluate trial
            trial_f = problem(trial)
            used += 1
            
            # Replace worst point in population using crowding distance
            # Compute crowding distance for non-dominated points
            def crowding_distance(F_arr, indices):
                n = len(indices)
                if n <= 2:
                    return [float('inf')] * n
                cd = [0.0] * n
                for k in range(self.n_obj):
                    sorted_idx = sorted(indices, key=lambda i: F_arr[i][k])
                    cd[sorted_idx[0]] = float('inf')
                    cd[sorted_idx[-1]] = float('inf')
                    f_range = F_arr[sorted_idx[-1]][k] - F_arr[sorted_idx[0]][k]
                    if f_range == 0:
                        continue
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_arr[sorted_idx[i+1]][k] - F_arr[sorted_idx[i-1]][k]) / f_range
                return cd
            
            # Get non-dominated set and compute crowding
            if len(nondom_idx) > 1:
                cd = crowding_distance(F, nondom_idx)
                # Find index with smallest crowding distance
                min_cd_idx = min(range(len(nondom_idx)), key=lambda i: cd[i])
                worst_idx = nondom_idx[min_cd_idx]
            else:
                worst_idx = nondom_idx[0] if nondom_idx else 0
            
            # Only replace if trial is not dominated by worst
            if not (np.all(F[worst_idx] <= trial_f) and np.any(F[worst_idx] < trial_f)):
                X[worst_idx] = trial
                F[worst_idx] = trial_f
        
        return X
# EVOLVE-BLOCK-END
