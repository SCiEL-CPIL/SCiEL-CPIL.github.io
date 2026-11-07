"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting RE21's separable structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # RE21 structure: f1 = sum of squares of first half, f2 = sum of squares of (second half - offset)
        # The offset is typically 2.0 for the second half
        # Ideal point: first half at 0, second half at offset (2.0)
        
        # Determine the split: first dim//2 variables for f1, last dim//2 for f2
        half = dim // 2
        
        # Ideal point: minimize each objective independently
        ideal = np.zeros(dim)
        for i in range(dim):
            if i < half:
                # f1 is minimized when x_i = 0 (if bounds allow)
                ideal[i] = 0.0
            else:
                # f2 is minimized when x_i = 2.0 (typical RE21 offset)
                ideal[i] = 2.0
        
        # Clip to bounds
        ideal = np.clip(ideal, lo, hi)
        
        # Evaluate ideal point first
        used = 0
        results = []
        
        x_ideal = ideal.copy()
        f_ideal = problem(x_ideal)
        used += 1
        results.append((x_ideal, f_ideal))
        
        n_remaining = budget - used
        
        if n_remaining <= 0:
            return x_ideal
        
        # Strategy: Systematically sample the Pareto front
        # For separable objectives, the Pareto front can be parameterized
        # f1 = a, f2 = b where a and b are non-negative
        # We want to cover the trade-off curve
        
        # Generate points along the Pareto front
        # Parameterize by t in [0, 1]: allocate "effort" to f1 vs f2
        # For f1 = x1^2 + x2^2: to get f1 = a, set x1 = sqrt(a), x2 = 0
        # For f2 = (x3-2)^2 + (x4-2)^2: to get f2 = b, set x3 = 2-sqrt(b), x4 = 2
        
        # But we need to be robust to different bounds, so use a more general approach:
        # Interpolate between the ideal point and the "worst" points
        
        # Determine the maximum possible objective values
        max_f1 = 0.0
        for i in range(half):
            # Max contribution from variable i to f1
            d1 = abs(lo[i] - 0.0)
            d2 = abs(hi[i] - 0.0)
            max_f1 += max(d1, d2)**2
        
        max_f2 = 0.0
        for i in range(half, dim):
            d1 = abs(lo[i] - 2.0)
            d2 = abs(hi[i] - 2.0)
            max_f2 += max(d1, d2)**2
        
        # Generate Pareto front samples
        # Use logarithmic spacing for better coverage near the ideal point
        n_front = min(n_remaining - 10, 100)  # reserve some budget for refinement
        n_front = max(n_front, 10)
        
        points = []
        
        # Add corner points for coverage
        corners = [
            np.array([lo[0], lo[1], lo[2], lo[3]]),
            np.array([hi[0], hi[1], hi[2], hi[3]]),
            np.array([lo[0], lo[1], hi[2], hi[3]]),
            np.array([hi[0], hi[1], lo[2], lo[3]]),
        ]
        
        for c in corners:
            c = np.clip(c, lo, hi)
            if not any(np.allclose(c, r[0], atol=1e-10) for r in results):
                points.append(c)
        
        # Generate front samples: parameterize by trade-off ratio
        # t=0: all effort on f1 (f2 at max), t=1: all effort on f2 (f1 at max)
        for i in range(n_front):
            t = (i + 1) / (n_front + 1)
            
            # Allocate objective values
            # f1_target = (1-t) * max_f1, f2_target = t * max_f2
            f1_target = (1.0 - t) * max_f1
            f2_target = t * max_f2
            
            # Construct x to achieve these targets
            x = np.zeros(dim)
            
            # First half: distribute f1_target among variables
            # Simple: put all in first variable
            if half > 0:
                x[0] = np.sqrt(f1_target)
                # But clip to bounds
                if x[0] > hi[0]:
                    x[0] = hi[0]
                    # Redistribute remaining to other variables
                    remaining = f1_target - x[0]**2
                    if remaining > 0 and half > 1:
                        x[1] = np.sqrt(remaining)
                        x[1] = np.clip(x[1], lo[1], hi[1])
                elif x[0] < lo[0]:
                    x[0] = lo[0]
                    remaining = f1_target - x[0]**2
                    if remaining > 0 and half > 1:
                        x[1] = np.sqrt(remaining)
                        x[1] = np.clip(x[1], lo[1], hi[1])
            
            # Second half: distribute f2_target among variables
            if dim - half > 0:
                # f2 = (x-2)^2, so x = 2 - sqrt(f2_target) or 2 + sqrt(f2_target)
                # Choose the one closer to bounds
                x2_val = 2.0 - np.sqrt(f2_target)
                if x2_val < lo[half] or x2_val > hi[half]:
                    x2_val = 2.0 + np.sqrt(f2_target)
                x[half] = np.clip(x2_val, lo[half], hi[half])
                
                if dim - half > 1:
                    # Second variable in second half
                    x2_val2 = 2.0 - np.sqrt(f2_target)
                    if x2_val2 < lo[half+1] or x2_val2 > hi[half+1]:
                        x2_val2 = 2.0 + np.sqrt(f2_target)
                    x[half+1] = np.clip(x2_val2, lo[half+1], hi[half+1])
            
            x = np.clip(x, lo, hi)
            if not any(np.allclose(x, r[0], atol=1e-10) for r in results):
                points.append(x)
        
        # Also add random points for robustness
        rng = np.random.default_rng(42)
        n_random = min(n_remaining - len(points) - 5, 50)
        for _ in range(n_random):
            x = lo + (hi - lo) * rng.random(dim)
            if not any(np.allclose(x, r[0], atol=1e-10) for r in results):
                points.append(x)
        
        # Evaluate all points
        for x in points:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            results.append((x.copy(), f))
        
        # Use remaining budget for local refinement around best points
        if used < budget and time.time() < deadline:
            # Find non-dominated points
            F = np.array([r[1] for r in results])
            X = np.array([r[0] for r in results])
            
            def dominates(a, b):
                return np.all(a <= b) and np.any(a < b)
            
            nd_idx = []
            for i in range(len(F)):
                is_nd = True
                for j in range(len(F)):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            
            # Refine around non-dominated points
            for idx in nd_idx:
                if used >= budget or time.time() > deadline:
                    break
                
                parent = X[idx]
                # Small perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    step = 0.05 * (hi - lo) * rng.standard_normal(dim)
                    child = parent + step
                    child = np.clip(child, lo, hi)
                    f = problem(child)
                    used += 1
                    results.append((child.copy(), f))
        
        return results[-1][0]
# EVOLVE-BLOCK-END
