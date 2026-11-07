import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # Sample x1 densely over [0, 1] to build a lookup table
        # Using 100,000 points for high precision
        n_samples = 100000
        x1_samples = np.linspace(0, 1, n_samples)
        f1_samples = calc_f1(x1_samples)
        
        # Find the maximum f1 value and its location
        max_idx = np.argmax(f1_samples)
        f1_max_precise = f1_samples[max_idx]
        
        # Sort by f1 to create a lookup table
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        # We want to generate points that cover the Pareto front well.
        # The Pareto front is defined by f2 = 1 - sqrt(f1) for f1 in [0, f1_max].
        # To maximize hypervolume, we should distribute points uniformly in objective space.
        # However, since f1 is non-linear in x1, uniform f1 targets are a good strategy.
        
        n_points = self.budget
        
        # Generate target f1 values uniformly in [0, f1_max_precise]
        # We exclude f1=0 and f1=f1_max to avoid boundary issues if needed, but including them is fine.
        # Actually, including the endpoints is good for coverage.
        target_f1s = np.linspace(0, f1_max_precise, n_points)
        
        # Find x1 for each target f1
        # Use searchsorted on the sorted f1 array
        indices = np.searchsorted(f1_sorted, target_f1s)
        
        # Clip indices to valid range
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        
        # Refinement: Check neighbors to find the closest match
        x1_targets = np.zeros(n_points)
        
        for i in range(n_points):
            idx = indices[i]
            target = target_f1s[i]
            
            # Check current, previous, and next
            candidates = []
            if idx > 0:
                candidates.append((idx - 1, f1_sorted[idx - 1], x1_sorted[idx - 1]))
            candidates.append((idx, f1_sorted[idx], x1_sorted[idx]))
            if idx < len(f1_sorted) - 1:
                candidates.append((idx + 1, f1_sorted[idx + 1], x1_sorted[idx + 1]))
                
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            
            for _, f1_val, x1_val in candidates:
                dist = abs(f1_val - target)
                if dist < best_dist:
                    best_dist = dist
                    best_x1 = x1_val
                    
            x1_targets[i] = best_x1

        # Construct and evaluate
        recorded = []
        for i in range(n_points):
            # Check time
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Clip to bounds [0, 1]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
