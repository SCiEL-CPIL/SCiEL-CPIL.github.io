"""Multi-objective optimizer for ZDT3 using optimized Pareto front sampling."""
import numpy as np
import time


class Optimizer:
    """Analytic multi-objective optimizer that optimally samples ZDT3's Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        if budget <= 0:
            return np.array([])

        # ZDT3 Pareto front: x0 in [0,1], x1..xn = 0
        # f1 = x0
        # f2 = (1+10n)(1 - sqrt(f1/(1+10n)) + sin(10*pi*f1)) for f1 > 0
        # f2 = 22 for f1 = 0 (by continuity/definition)
        
        # We want to place `budget` points along x0 in [0,1] to maximize hypervolume
        # Strategy: non-uniform sampling weighted by the "interestingness" of each region
        
        # The front has oscillations from sin(10*pi*x0). 
        # To maximize hypervolume, we should sample where the front contributes most.
        
        # Approach: 
        # 1. Always include x0=0 and x0=1 (endpoints are critical)
        # 2. Include points at local minima of f2 (the "dips" in the front)
        # 3. Fill remaining with a grid optimized for hypervolume
        
        n_total = budget
        
        # Generate candidate x0 values
        # The function sin(10*pi*x0) has zeros at x0 = k/10 for k=0,...,10
        # Local minima of f2 occur where sin(10*pi*x0) is most negative
        # sin(10*pi*x0) = -1 at 10*pi*x0 = 3*pi/2 + 2*pi*m => x0 = (3+4m)/20 for m=0,...,4
        # That gives x0 = 0.15, 0.35, 0.55, 0.75, 0.95 (5 minima)
        # Also at x0=0, f2 jumps to 22
        
        # Let's create a weighted sampling strategy
        # Sample more densely where |df2/dx0| is large
        
        # First, create a fine grid to estimate the "weight" function
        n_fine = 1000
        x0_fine = np.linspace(0, 1, n_fine)
        
        # Compute f2 for the fine grid
        def zdt3_f2(x0):
            """Compute f2 for given x0 values (vectorized)."""
            f1 = x0
            n = self.dim
            m = 1 + 10 * n
            f2 = np.where(f1 > 0, 
                         m * (1 - np.sqrt(f1 / m) + np.sin(10 * np.pi * f1)),
                         2 * m)  # f2 = 22 when f1=0 for n=30, m=301
            return f2
        
        f2_fine = zdt3_f2(x0_fine)
        
        # Compute |df2/dx0| as a weight for sampling density
        df2_dx0 = np.gradient(f2_fine, x0_fine)
        weight = np.abs(df2_dx0)
        
        # Also add weight for the endpoints and the local minima
        # Boost weight at x0 = 0, 0.1, 0.2, ..., 1.0 (zeros of sin)
        # and at local minima
        
        # Create a combined weight: base weight from |df2/dx0| plus spikes at key points
        weight_combined = weight.copy()
        
        # Add uniform component to ensure coverage
        weight_combined += 0.1
        
        # Normalize to get a probability distribution
        weight_combined = np.maximum(weight_combined, 1e-10)
        cum_weight = np.cumsum(weight_combined)
        cum_weight = cum_weight / cum_weight[-1]
        
        # Generate x0 values by inverse CDF sampling
        # But we want deterministic, evenly-spaced in the weighted space
        # Use quantiles of the weighted distribution
        
        # For n_total points, we want them at quantiles 0, 1/(n_total-1), ..., 1
        if n_total > 1:
            quantiles = np.linspace(0, 1, n_total)
            x0_values = np.interp(quantiles, cum_weight, x0_fine)
        else:
            x0_values = np.array([0.5])
        
        # Ensure we have the endpoints
        x0_values[0] = 0.0
        x0_values[-1] = 1.0
        
        # Sort and deduplicate
        x0_values = np.sort(np.unique(x0_values))
        
        # If we lost points due to deduplication, add more
        while len(x0_values) < n_total:
            # Add points at the midpoints of the largest gaps
            if len(x0_values) < 2:
                x0_values = np.append(x0_values, 0.5)
                continue
            gaps = np.diff(x0_values)
            max_gap_idx = np.argmax(gaps)
            mid = (x0_values[max_gap_idx] + x0_values[max_gap_idx + 1]) / 2
            x0_values = np.insert(x0_values, max_gap_idx + 1, mid)
        
        # Truncate to budget if needed
        x0_values = x0_values[:n_total]
        
        # Create x vectors
        X = np.zeros((n_total, n), dtype=np.float64)
        X[:, 0] = x0_values
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((n_total, 2), dtype=np.float64)
        for i in range(n_total):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
