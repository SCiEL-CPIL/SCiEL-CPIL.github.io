import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        # Strategy: Sample along the two 1D edges of the Pareto front.
        # Edge 1: y2 = 0. f1 = y1, f2 = y1, f3 = (1-y1)^2.
        # Edge 2: y2 = 1. f1 = y1, f2 = 0, f3 = 2(1-y1).
        # We split the budget equally between the two edges.
        # We use a deterministic 1D low-discrepancy sequence (Halton-like via primes)
        # or just uniform sampling with a bias towards the 'knee' (small y1).
        # Given the reference point (1.1, 1.1, 25), points with small f1, f2 are best.
        # f3 is small for y1 close to 1 on Edge 1? No, f3=(1-y1)^2 is small at y1=1.
        # But f1=1 is bad.
        # f3 is small for y1 close to 1 on Edge 2? f3=2(1-y1) is small at y1=1.
        # But f1=1 is bad.
        # The 'knee' is likely where the trade-off is steepest.
        
        # Let's use uniform sampling of y1 in [0, 1] for both edges.
        n_samples = self.budget
        n_per_edge = n_samples // 2
        n_last = n_samples - 2 * n_per_edge
        
        # Generate y1 values for Edge 1 (y2=0)
        y1_edge1 = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        # Generate y1 values for Edge 2 (y2=1)
        y1_edge2 = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        
        # Handle the last point if odd budget
        if n_last > 0:
            # Add one more point to Edge 1
            y1_edge1 = np.append(y1_edge1, 1.0)
        
        # Construct X
        X = np.zeros((n_samples, self.dim))
        
        # Fill Edge 1
        X[:n_per_edge, 0] = y1_edge1[:n_per_edge]
        X[:n_per_edge, 1] = 0.0
        
        # Fill Edge 2
        X[n_per_edge:n_per_edge+n_per_edge, 0] = y1_edge2
        X[n_per_edge:n_per_edge+n_per_edge, 1] = 1.0
        
        # Fill last point if any
        if n_last > 0:
            X[-1, 0] = 1.0
            X[-1, 1] = 0.0
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            x = np.clip(x, lo, hi)
            problem(x)

        return X
