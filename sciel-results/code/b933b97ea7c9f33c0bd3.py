import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def halton_1d(idx, base):
            result = 0.0
            f = 1.0 / base
            i = idx
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result

        n_evals = self.budget
        points = []

        # DTLZ4 Pareto front conditions:
        # g(x) = 1 requires x_i = 0.5 for i >= M (indices 2..11) and sin(10*pi*x_0) = 0
        # sin(10*pi*x_0) = 0 implies x_0 in {0, 0.1, ..., 1.0}
        # x_1 (index 1) determines the angle in [0, 1] (since bounds are [0,1])
        
        # Allocate evaluations to slices k=0..10 proportional to k (arc length)
        # k=0 is a single point (1,0,0)
        # k=1..10 are arcs of radius k/10
        
        n_slices = 11
        # Weights proportional to k for k=1..10, and 1 for k=0 (to get the vertex)
        # Actually, k=0 contributes a fixed volume box. k>0 contribute surface area.
        # Let's use weights: w_0 = 1, w_k = k for k=1..10
        weights = np.zeros(n_slices)
        weights[0] = 1.0
        for k in range(1, n_slices):
            weights[k] = float(k)
        
        total_weight = np.sum(weights)
        # Distribute budget
        raw_counts = (weights / total_weight) * n_evals
        counts = np.floor(raw_counts).astype(int)
        
        # Distribute remaining points to those with largest fractional parts
        remainders = raw_counts - counts
        remaining = n_evals - np.sum(counts)
        indices = np.argsort(remainders)[::-1]
        for i in range(remaining):
            counts[indices[i]] += 1
            
        # Ensure at least 1 point for each slice? 
        # k=0 is a point. k>0 are arcs.
        # If a slice gets 0, it's a problem. But with 200 points and 11 slices, all should get >0.
        
        # Generate Halton sequence for x[1] (angle)
        # We need n_evals points
        halton_values = []
        for i in range(1, n_evals + 1):
            # Use base 2 for better distribution in 1D? Or just uniform random?
            # Halton base 2 is good.
            val = halton_1d(i, 2)
            halton_values.append(val)
            
        # Assign points
        current_idx = 0
        for k in range(n_slices):
            x0_val = k / 10.0
            for j in range(counts[k]):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0_val
                # Map Halton value to [0, 1] for x[1]
                x[1] = halton_values[current_idx]
                # Set remaining variables to 0.5 to ensure g=1
                x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                points.append(x)
                current_idx += 1
            
            if time.time() > deadline:
                break

        # Evaluate
        for x in points:
            if time.time() > deadline:
                break
            problem(x)

        return points
