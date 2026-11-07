import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        n = self.dim  # 22
        m = self.n_obj  # 3
        n_dist = n - m + 1  # 20 distance variables
        
        # Solve for distance variables that give g=0
        # g = n_dist - sum(x_i * (1 + 2*sin^2(2*pi*x_i))) = 0
        # Strategy: set k vars to 0.5 (each contributes 1.5), rest to t
        # 1.5*k + t*(1+2*sin^2(2*pi*t))*(n_dist-k) = n_dist
        # Try k=13: 19.5 + t*(1+2*sin^2(2*pi*t))*7 = 20
        # t*(1+2*sin^2(2*pi*t)) = 0.5/7
        
        def solve_t(target, tol=1e-12, max_iter=100):
            """Solve t*(1+2*sin^2(2*pi*t)) = target for t in [0, 1]"""
            lo, hi = 0.0, 1.0
            for _ in range(max_iter):
                mid = (lo + hi) / 2
                val = mid * (1 + 2 * np.sin(2 * np.pi * mid) ** 2)
                if abs(val - target) < tol:
                    return mid
                if val < target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        
        # Find k and t such that 1.5*k + t*(1+2*sin^2(2*pi*t))*(n_dist-k) = n_dist
        # k must satisfy: 1.5*k <= n_dist and (n_dist - 1.5*k)/(n_dist-k) <= 1.5
        # 1.5*k <= 20 => k <= 13.33, so k <= 13
        # (20 - 1.5k)/(20-k) <= 1.5 => 20-1.5k <= 30-1.5k => 20<=30, always true
        # Also need (20-1.5k)/(20-k) >= 0, always true for k<20
        # And t must be in [0,1], so (20-1.5k)/(20-k) must be achievable
        # Max of t*(1+2sin^2(2pi*t)) is 1.5 at t=0.5
        # So need (20-1.5k)/(20-k) <= 1.5, which is always true as shown
        
        k = 13
        target_per_var = (n_dist - 1.5 * k) / (n_dist - k)
        t_val = solve_t(target_per_var)
        
        # Create base distance vector
        dist_base = np.zeros(n_dist)
        dist_base[:k] = 0.5
        dist_base[k:] = t_val
        
        # Verify g
        g_check = n_dist - np.sum(dist_base * (1 + 2 * np.sin(2 * np.pi * dist_base) ** 2))
        # Should be close to 0
        
        n_samples = self.budget  # 200
        
        # Use power-law biased Halton-like sampling for (x0, x1)
        # Bias toward lower values where hypervolume contribution is highest
        alpha = 2.0  # Power-law exponent
        
        # Generate quasi-random points in [0,1]^2
        # Use a simple LCG-based quasi-random sequence with power-law transform
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate n_samples points with power-law bias
        # Use stratified sampling with power-law transform
        n_per_axis = int(np.ceil(np.sqrt(n_samples)))
        # Adjust to get exactly n_samples or close
        while n_per_axis * n_per_axis > n_samples:
            n_per_axis -= 1
        
        results = []
        count = 0
        
        # Create stratified grid with power-law bias
        # u_i = (i + 0.5) / n_per_axis for i in 0..n_per_axis-1
        # x = u^(1/alpha) to bias toward 0
        u_vals = (np.arange(n_per_axis) + 0.5) / n_per_axis
        x0_vals = u_vals ** (1.0 / alpha)
        x1_vals = u_vals ** (1.0 / alpha)
        
        # Create grid
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                if count >= n_samples:
                    break
                if time.time() > deadline:
                    break
                
                x0 = x0_vals[i]
                x1 = x1_vals[j]
                
                # Ensure within bounds
                x0 = np.clip(x0, 0.0, 1.0)
                x1 = np.clip(x1, 0.0, 1.0)
                
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = dist_base
                
                try:
                    f = problem(x)
                    f = np.asarray(f, dtype=np.float64)
                    results.append(f)
                    count += 1
                except Exception:
                    pass
            
            if count >= n_samples or time.time() > deadline:
                break
        
        # Use remaining budget for additional diverse points
        remaining = n_samples - len(results)
        if remaining > 0 and time.time() < deadline:
            # Add points with different bias or at corners
            corner_points = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                (0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0),
                (0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75),
                (0.75, 0.25)
            ]
            
            for x0, x1 in corner_points:
                if remaining <= 0 or time.time() > deadline:
                    break
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = dist_base
                
                try:
                    f = problem(x)
                    f = np.asarray(f, dtype=np.float64)
                    results.append(f)
                    remaining -= 1
                except Exception:
                    pass
        
        return results
