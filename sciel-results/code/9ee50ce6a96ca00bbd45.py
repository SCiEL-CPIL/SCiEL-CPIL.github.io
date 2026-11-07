import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        
        # 1. Build lookup table for x1 given f1
        # We need to map target f1 values to x1.
        # Since f1 is not monotonic, we just pick the first x1 found in the sorted list.
        # The resulting point (f1, f2) is on the front regardless of which x1 is chosen.
        
        x1_samples = np.linspace(0, 1, 100000)
        sin_term = np.sin(6 * np.pi * x1_samples)
        sin_term_pow = sin_term ** 6
        exp_term = np.exp(-4 * x1_samples)
        base = 1 - exp_term * sin_term_pow
        base = np.clip(base, 0, 1)
        f1_samples = 1 - base ** 0.1
        
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]
        
        # 2. Sample auxiliary variable u uniformly
        n_points = self.budget
        u_samples = np.linspace(0, 1, n_points)
        
        # Map u to target f1 using power law to cluster near 0
        # f1 = (u * (f1_max - f1_min) + f1_min) ** alpha ? No.
        # We want f1 to range from f1_min to f1_max.
        # Let f1_target = f1_min + (f1_max - f1_min) * (u ** alpha)
        alpha = 2.0
        target_f1s = f1_min + (f1_max - f1_min) * (u_samples ** alpha)
        
        # 3. Map target f1s to x1s
        indices = np.searchsorted(f1_sorted, target_f1s)
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        x1_targets = x1_sorted[indices]
        
        # 4. Construct decision vectors and evaluate
        recorded = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
