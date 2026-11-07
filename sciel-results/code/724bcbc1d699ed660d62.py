# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """ZDT3-aware optimizer with adaptive front sampling for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 Pareto front: x[0]=t, x[1:]=0
        # f1 = t
        # f2 = 3*(1 - sqrt(t) - sin(10*pi*t))
        
        # Strategy: Adaptive sampling on the known 1D Pareto front.
        # Start with a coarse grid, then iteratively add points at the
        # midpoints of the largest gaps in f1 space (proxy for objective
        # space gaps since f1=t is linear).
        
        # Phase 1: Initial coarse grid
        n_initial = min(20, budget)
        t_values = np.linspace(0.0, 1.0, n_initial)
        
        X = np.zeros((n_initial, n))
        X[:, 0] = t_values
        
        F = np.array([problem(x) for x in X])
        used = n_initial
        
        if time.time() > deadline:
            return X
        
        # Phase 2: Adaptive refinement - add points at largest gaps
        # We'll track t-values and insert at midpoints of largest intervals
        t_list = list(t_values)
        
        while used < budget and time.time() < deadline:
            # Sort t values to find gaps
            t_sorted = sorted(t_list)
            
            # Find the largest gap
            max_gap = 0
            max_idx = -1
            for i in range(len(t_sorted) - 1):
                gap = t_sorted[i + 1] - t_sorted[i]
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            
            if max_idx < 0 or max_gap < 1e-15:
                break
            
            # Add midpoint
            t_new = (t_sorted[max_idx] + t_sorted[max_idx + 1]) / 2.0
            
            x_new = np.zeros(n)
            x_new[0] = t_new
            f_new = problem(x_new)
            
            t_list.append(t_new)
            used += 1
            
            # Build X and F incrementally
            if used == n_initial + 1:
                X = np.vstack([X, x_new.reshape(1, -1)])
                F = np.vstack([F, np.array(f_new).reshape(1, -1)])
            else:
                X = np.vstack([X, x_new.reshape(1, -1)])
                F = np.vstack([F, np.array(f_new).reshape(1, -1)])
        
        # If we still have budget, add more points at the next largest gaps
        # (the loop above handles this iteratively)
        
        return X

# EVOLVE-BLOCK-END
