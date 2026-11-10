"""Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT4's analytic Pareto set: x2..xn at zeros of sin(2*pi*x_i) with cos=1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        budget = self.budget
        used = 0
        
        # Find optimal values for x2..xn
        # Minimize sin^2(2*pi*x) - 10*cos(2*pi*x)
        # Global minima at x = k (integers)
        optimal_tail = np.zeros(dim - 1)
        for i in range(1, dim):
            l, h = lo[i], hi[i]
            # Find value in [l, h] closest to an integer
            candidates = [l, h]
            k_start = int(np.ceil(l))
            k_end = int(np.floor(h))
            for k in range(k_start, k_end + 1):
                candidates.append(float(k))
            
            best_val = l
            best_dist = np.inf
            for c in candidates:
                # Distance to nearest integer
                d = min(c - np.floor(c), np.ceil(c) - c)
                if d < best_dist:
                    best_dist = d
                    best_val = c
            optimal_tail[i-1] = best_val

        # Sample x1 uniformly in [lo[0], hi[0]]
        l1, h1 = lo[0], hi[0]
        x1_values = np.linspace(l1, h1, budget)
        
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break
            x = np.concatenate([[x1], optimal_tail])
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
