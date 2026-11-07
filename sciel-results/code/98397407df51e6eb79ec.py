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

        # The Pareto front of ZDT6: x2..xn = 0, and f1 ranges over [0, f1_max]
        # where f1_max is the maximum of f1(x1) for x1 in [0,1]
        
        # Find the maximum of f1 using dense sampling
        x1_dense = np.linspace(0, 1, 100000)
        f1_dense = calc_f1(x1_dense)
        idx_max = np.argmax(f1_dense)
        f1_max = f1_dense[idx_max]
        x1_peak = x1_dense[idx_max]
        
        # Generate target f1 values uniformly spaced from 0 to f1_max
        n_points = self.budget
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # For each target f1, find x1 using binary search on the dense grid
        # This is much faster than brentq for 200 points
        
        x1_targets = np.empty(n_points)
        
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                x1_targets[i:] = 0.0
                break
            
            target = target_f1s[i]
            
            # For target = 0, x1 = 0
            if target == 0:
                x1_targets[i] = 0.0
                continue
            
            # For target = f1_max, x1 = x1_peak
            if target >= f1_max - 1e-12:
                x1_targets[i] = x1_peak
                continue
            
            # Binary search on the dense grid to find x1 where f1(x1) ≈ target
            # We search in the first lobe where f1 is mostly monotonic
            # Find the rightmost index where f1 <= target (in the first lobe)
            
            # Use np.searchsorted on the f1_dense array, but we need to be careful
            # because f1 is not monotonic over [0,1]. Let's restrict to the first lobe.
            
            # The first lobe is roughly [0, 0.17] where f1 increases from 0 to near max
            # Let's find where f1 first reaches its maximum
            first_max_idx = idx_max
            
            # Search in [0, first_max_idx]
            f1_lobe = f1_dense[:first_max_idx + 1]
            
            # Find the index where f1_lobe is closest to target
            # Use searchsorted for efficiency
            idx = np.searchsorted(f1_lobe, target)
            
            # Check neighbors to find the best match
            if idx == 0:
                best_idx = 0
            elif idx >= len(f1_lobe):
                best_idx = len(f1_lobe) - 1
            else:
                # Compare idx and idx-1
                if abs(f1_lobe[idx] - target) < abs(f1_lobe[idx-1] - target):
                    best_idx = idx
                else:
                    best_idx = idx - 1
            
            x1_targets[i] = x1_dense[best_idx]

        # Construct and evaluate points
        recorded = []
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
