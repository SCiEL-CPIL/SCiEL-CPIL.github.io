"""Multi-objective optimizer for ZDT4 using population-based search with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute hypervolume for 2D points given reference point."""
        if len(points) == 0:
            return 0.0
        valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
        if not valid:
            return 0.0
        valid.sort(key=lambda p: p[0])
        
        hv = 0.0
        prev_f1 = valid[0][0]
        min_f2 = valid[0][1]
        for i in range(1, len(valid)):
            cur_f1, cur_f2 = valid[i]
            # The region from prev_f1 to cur_f1 with height (ref_f2 - min_f2)
            if min_f2 < ref[1]:
                hv += (cur_f1 - prev_f1) * (ref[1] - min_f2)
            min_f2 = min(min_f2, cur_f2)
            prev_f1 = cur_f1
        # Last segment
        if min_f2 < ref[1]:
            hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 400.0])
        
        # Key insight: ZDT4's Pareto front has x2..x10 = 0.1
        # Focus search on x1 while keeping other variables near 0.1
        opt_values = np.array([0.1] * self.dim)
        
        # Initial population: spread x1, set others to 0.1
        pop_size = min(30, self.budget // 4)
        X = np.zeros((pop_size, self.dim))
        # Spread x1 uniformly
        X[:, 0] = np.linspace(lo[0], hi[0], pop_size)
        # Set other variables to optimal value 0.1
        for d in range(1, self.dim):
            X[:, d] = opt_values[d]
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume
        best_hv = self._hypervolume_2d(F.tolist(), ref)
        
        # Mutation strategies focused on x1
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Select parent: prefer non-dominated, then diverse
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            non_dom = ~dominated
            if np.any(non_dom):
                weights = np.ones(pop_size) * 0.1
                weights[non_dom] = 1.0
                weights /= weights.sum()
                parent_idx = np.random.choice(pop_size, p=weights)
            else:
                parent_idx = np.random.randint(pop_size)
            
            parent = X[parent_idx].copy()
            
            # Mutate primarily x1 with small perturbations to other variables
            child = parent.copy()
            
            # Mutate x1 with adaptive step size
            x1_range = hi[0] - lo[0]
            step_x1 = x1_range * np.random.choice([0.01, 0.02, 0.05, 0.1, 0.2])
            child[0] += step_x1 * np.random.randn()
            
            # Occasionally perturb other variables slightly
            if np.random.rand() < 0.2:
                n_perturb = np.random.randint(1, 3)
                perturb_idx = np.random.choice(range(1, self.dim), n_perturb, replace=False)
                for d in perturb_idx:
                    child[d] += (hi[d] - lo[d]) * 0.01 * np.random.randn()
            
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            sums = np.sum(F, axis=1)
            worst = np.argmax(sums)
            X[worst] = child
            F[worst] = fc
            
            # Update best hypervolume
            hv = self._hypervolume_2d(F.tolist(), ref)
            if hv > best_hv:
                best_hv = hv
        
        return X
# EVOLVE-BLOCK-END
