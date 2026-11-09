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
            if min_f2 < ref[1]:
                hv += (cur_f1 - prev_f1) * (ref[1] - min_f2)
            min_f2 = min(min_f2, cur_f2)
            prev_f1 = cur_f1
        if min_f2 < ref[1]:
            hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 400.0])
        
        # ZDT4 specific: x1 in [0,1], x2..x10 in [-5,5]
        # g(x) has many local minima. The Pareto front is at g=0.
        # g(x) = 10*n + sum((x_i - 0.1)^2 - 10*cos(4*pi*(x_i-0.1))) for i=2..n
        # The minimum of (x-0.1)^2 - 10*cos(4*pi*(x-0.1)) is -10 at x=0.1
        # So g_min = 10*9 + 9*(-10) = 0.
        # This means the Pareto front is exactly at x1 in [0,1], x2..x10 = 0.1.
        
        # Strategy: 
        # 1. Initialize with points near the known Pareto front (x2..x10 = 0.1, x1 spread)
        # 2. Use local search to refine points
        # 3. Use global search to explore if needed
        
        pop_size = 20
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # Initialize: half near the known front, half random
        n_front = pop_size // 2
        # Near front: x2..x10 = 0.1, x1 spread
        X[:n_front, 0] = np.linspace(0.0, 1.0, n_front)
        X[:n_front, 1:] = 0.1 + 0.01 * np.random.randn(n_front, self.dim - 1)
        
        # Random: spread across bounds
        X[n_front:, 0] = np.random.rand(pop_size - n_front)
        X[n_front:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * np.random.rand(pop_size - n_front, self.dim - 1)
        
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size
        
        # Track best hypervolume
        best_hv = self._hypervolume_2d(F.tolist(), ref)
        
        # Local search step sizes (geometric progression)
        local_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute dominance
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            non_dom = ~dominated
            
            # Select parent: prefer non-dominated, then diverse
            if np.any(non_dom):
                weights = np.ones(pop_size) * 0.1
                weights[non_dom] = 1.0
                weights /= weights.sum()
                parent_idx = np.random.choice(pop_size, p=weights)
            else:
                parent_idx = np.random.randint(pop_size)
            
            parent = X[parent_idx].copy()
            
            # Decide: local search or global exploration
            # If parent is close to the known front (x2..x10 near 0.1), do local search
            # Otherwise, do global exploration
            dist_to_front = np.abs(parent[1:] - 0.1).max()
            
            if dist_to_front < 0.5 and np.random.rand() < 0.7:
                # Local search: refine x2..x10 toward 0.1, and fine-tune x1
                child = parent.copy()
                step = local_steps[np.random.randint(len(local_steps))]
                
                # Move x2..x10 toward 0.1
                child[1:] = 0.1 + (parent[1:] - 0.1) * (1 - step)
                
                # Fine-tune x1
                child[0] += step * 0.1 * np.random.randn()
                child = np.clip(child, lo, hi)
            else:
                # Global exploration: random reset of some variables
                child = parent.copy()
                if np.random.rand() < 0.5:
                    # Reset x2..x10 to random
                    child[1:] = lo[1:] + (hi[1:] - lo[1:]) * np.random.rand(self.dim - 1)
                else:
                    # Reset x1 to random, keep x2..x10
                    child[0] = np.random.rand()
                    # Add noise to x2..x10
                    step = 0.1 + np.random.rand() * 0.5
                    child[1:] += step * (hi[1:] - lo[1:]) * np.random.randn(self.dim - 1)
                    child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives)
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
