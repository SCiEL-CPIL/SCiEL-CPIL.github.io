import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        # ZDT3 has 7 disjoint Pareto front segments
        # Each segment corresponds to a specific configuration of x[1:] that achieves g=1
        # The segments are defined by the local minima of h(x) = 5x + sin(10x)
        
        # Find the 7 local minima of h(x) = 5x + sin(10x) on [0,1]
        # h'(x) = 5 + 10cos(10x) = 0 when cos(10x) = -0.5
        # 10x = 2π/3 + 2kπ or 10x = 4π/3 + 2kπ
        # x = (2π/3 + 2kπ)/10 or x = (4π/3 + 2kπ)/10
        
        # Local minima occur where h''(x) = -100sin(10x) > 0, i.e., sin(10x) < 0
        # For x = (2π/3 + 2kπ)/10: 10x = 2π/3 + 2kπ, sin(10x) = sin(2π/3) = √3/2 > 0 (maximum)
        # For x = (4π/3 + 2kπ)/10: 10x = 4π/3 + 2kπ, sin(10x) = sin(4π/3) = -√3/2 < 0 (minimum)
        
        # So local minima are at x = (4π/3 + 2kπ)/10 for k=0,1,...
        # In [0,1]: k=0 gives x≈0.4189, k=1 gives x≈1.0575 (out of range)
        # Wait, let me recalculate.
        
        # Actually, let's find all critical points in [0,1]:
        # 10x = 2π/3 + 2kπ → x = (2π/3 + 2kπ)/10
        # 10x = 4π/3 + 2kπ → x = (4π/3 + 2kπ)/10
        
        critical_points = []
        for k in range(0, 10):
            x1 = (2*np.pi/3 + 2*k*np.pi) / 10
            x2 = (4*np.pi/3 + 2*k*np.pi) / 10
            if 0 <= x1 <= 1:
                critical_points.append(x1)
            if 0 <= x2 <= 1:
                critical_points.append(x2)
        
        # Classify as min or max using second derivative
        # h''(x) = -100sin(10x)
        # If h''(x) > 0 (sin(10x) < 0), it's a minimum
        local_minima = []
        for x in critical_points:
            if np.sin(10*x) < 0:  # h''(x) > 0
                local_minima.append(x)
        
        # Also include x=0 and x=1 as boundary points
        local_minima.append(0.0)
        local_minima.append(1.0)
        local_minima = sorted(set(local_minima))
        
        # For ZDT3, the Pareto front segments correspond to x[1:] values where
        # the sum of h(x_i) achieves the minimum value for that segment
        # Each segment has a different number of variables at each local minimum
        
        # The 7 segments are defined by the number of variables set to each local minimum
        # to achieve g(x[1:]) = 1
        
        # For simplicity, let's assume the segments correspond to different combinations
        # of the local minima values for x[1:]
        
        # A simpler approach: sample x[1:] from the local minima values
        # and x[0] uniformly, then filter for points that achieve g≈1
        
        n_local_min = len(local_minima)
        
        # Phase 1: Explore to find the actual Pareto front structure
        n_explore = min(50, budget // 4)
        
        # Sample x[1:] using combinations of local minima
        explore_points = []
        for i in range(n_explore):
            x = np.zeros(dim)
            x[0] = np.random.uniform(lo[0], hi[0])
            # Randomly assign x[1:] from local minima
            x[1:] = np.random.choice(local_minima, size=dim-1)
            explore_points.append(x)
        
        f1_explore = np.zeros(n_explore)
        f2_explore = np.zeros(n_explore)
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            try:
                f = problem(explore_points[i])
                f1_explore[i] = f[0]
                f2_explore[i] = f[1]
            except Exception:
                break
        
        # Phase 2: Based on exploration, determine the best strategy
        # If we found points with low f2, use those x[1:] configurations
        valid = np.isfinite(f1_explore) & np.isfinite(f2_explore)
        if np.sum(valid) > 0:
            # Find the point with minimum f2
            min_f2_idx = np.argmin(f2_explore[valid])
            best_x1 = explore_points[valid][min_f2_idx][1:]
            
            # Phase 3: Use the best x[1:] configuration and sample x[0] precisely
            n_refine = budget - n_explore
            
            # Sample x[0] uniformly across the full range
            x0_samples = np.linspace(lo[0], hi[0], n_refine)
            
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0_samples[i]
                x[1:] = best_x1
                try:
                    problem(x)
                except Exception:
                    break
        else:
            # Fallback: random sampling
            n_refine = budget - n_explore
            for _ in range(n_refine):
                if time.time() > deadline:
                    break
                x = np.random.uniform(lo, hi)
                try:
                    problem(x)
                except Exception:
                    break
        
        return None
