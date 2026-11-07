"""ZDT6 optimizer using analytic Pareto front sampling in objective space."""
import numpy as np
import time


class Optimizer:
    """
    ZDT6-specific optimizer using analytic Pareto front sampling.
    
    Strategy:
    - ZDT6 has a known analytic Pareto front.
    - f1 = x1
    - g(x) = 1 + 10(n-1)/n + sum_{i=2}^n (x_i^2 - 2*x_1*cos(20*pi*x_i))
    - f2 = (1 - sqrt(f1/g(x))) * (1 - f1^2/100)
    
    The Pareto front is achieved when g(x) is minimized for each x1.
    For ZDT6, the minimum of g(x) for a fixed x1 occurs at specific x2..xn values.
    
    Key insight: We can directly compute points on the Pareto front by:
    1. Sampling x1 (which determines f1 = x1)
    2. For each x1, finding x2..xn that minimize g(x)
    
    However, finding the exact minimizer of g(x) is complex. Instead, we use
    the fact that for hypervolume scoring, we want points that are close to
    the true Pareto front. 
    
    A simpler and more effective approach: sample the Pareto front uniformly
    in objective space (f1, f2) rather than in x1 space. This ensures better
    coverage of the hypervolume.
    
    For ZDT6, when x1 is small, g(x) can be made close to its minimum by
    setting x2..xn appropriately. The key is that f1 = x1, so we can directly
    control f1.
    
    To get points on or very close to the Pareto front, we use the following
    approach:
    - Sample f1 values uniformly in [0, 1] (since f1 = x1)
    - For each f1, we need to find x2..xn that minimize g(x)
    - Since we can't easily solve for the exact minimizer, we use a local
      search approach with a small budget per point
    
    Actually, let me reconsider. The inspiration solutions score ~1.5, which
    suggests they're doing something clever. Let me think about what that could be.
    
    For ZDT6, the Pareto front is defined parametrically. If we can directly
    compute points on the front without needing to minimize g(x), that would be
    ideal. 
    
    Looking at the ZDT6 definition more carefully:
    - f1 = x1
    - To be on the Pareto front, we need g(x) to be at its minimum for the given x1
    
    The minimum of g(x) for a fixed x1 can be found by setting each x_i (i>=2)
    to minimize x_i^2 - 2*x_1*cos(20*pi*x_i). This is a 1D optimization problem
    for each variable.
    
    For small x1, the minimum is near x_i = 0. For larger x1, the cosine term
    creates multiple local minima.
    
    A practical approach: use the fact that we only need to be close to the
    Pareto front for good hypervolume. We can:
    1. Sample x1 uniformly in [0, 1]
    2. For each x1, use a few iterations of coordinate descent to minimize g(x)
       (by minimizing f2, since f1 is fixed)
    3. Record the resulting (f1, f2) points
    
    But even better: if we can analytically approximate the Pareto front, we
    can just sample it directly without any optimization.
    
    Let me use a hybrid approach:
    - Use most of the budget to sample points on an approximate Pareto front
    - Use a small amount of budget to refine the points via local search
    
    For the approximate front, I'll use the fact that for ZDT6, setting
    x2..xn to values that approximately minimize g(x) gives points close to
    the true front. A simple heuristic is to set x_i = 0.5 for all i >= 2,
    which doesn't minimize g(x) but gives reasonable f2 values.
    
    Actually, let me try a different approach inspired by the high-scoring
    solutions: sample the Pareto front uniformly in objective space.
    
    Since f1 = x1, we can directly control f1. For a given f1, the Pareto-optimal
    f2 is the minimum possible f2 for that f1, which requires minimizing g(x).
    
    I'll use the following strategy:
    1. Sample f1 values that are uniformly distributed in the range of f1
       on the Pareto front
    2. For each f1, use a local search to find x2..xn that minimize f2
    3. Record all resulting points
    
    To ensure good hypervolume, I'll sample f1 values that give uniform
    coverage in the (f1, f2) plane.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        # Strategy: Sample points on/near the Pareto front
        # f1 = x1, so we control f1 by setting x1
        # For each x1, minimize f2 by optimizing x2..xn
        
        # Use a mix of:
        # 1. Dense sampling of x1 values (which gives f1 values)
        # 2. Local search for x2..xn to minimize f2
        
        # Budget allocation:
        # - 80% for sampling x1 values with local search
        # - 20% for random exploration to catch any missed regions
        
        num_main_samples = 120  # Number of x1 values to sample
        remaining_budget = self.budget - num_main_samples
        if remaining_budget < 0:
            num_main_samples = self.budget
            remaining_budget = 0
            
        budget_per_sample = max(1, remaining_budget // num_main_samples)
        
        best_points = []
        
        # Sample x1 values - use non-uniform sampling to better cover the front
        # The Pareto front of ZDT6 is curved, so uniform in x1 may not be optimal
        # Use a power law to concentrate points where the front is steeper
        
        for i in range(num_main_samples):
            if time.time() > deadline - 5:
                break
                
            # x1 = f1, sample in [0, 1]
            # Use a slightly non-uniform distribution to better cover the front
            t = (i + 0.5) / num_main_samples
            x1_val = t ** 0.7  # Slightly concentrate near x1=0
            
            x = np.zeros(n)
            x[0] = x1_val
            
            # Initialize x2..xn - start with a reasonable guess
            # For ZDT6, setting x_i to values that approximately minimize
            # the cosine term can help
            x[1:] = 0.5 * (hi[1:] - lo[1:]) + lo[1:]
            
            f = problem(x)
            best_f = f.copy()
            best_x = x.copy()
            
            # Local search for x2..xn to minimize f2
            # Use a simple random perturbation approach
            steps = min(budget_per_sample, 5)  # Limit steps to avoid timeout
            for _ in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                # Perturb x2..xn with decreasing step size
                step_size = 0.1 / (1 + _)
                perturbation = step_size * (hi[1:] - lo[1:]) * np.random.standard_normal(n-1)
                x_new = best_x.copy()
                x_new[1:] += perturbation
                x_new[1:] = np.clip(x_new[1:], lo[1:], hi[1:])
                
                f_new = problem(x_new)
                
                # We want to minimize f2 (f1 is fixed by x1)
                if f_new[1] < best_f[1]:
                    best_f = f_new
                    best_x = x_new
            
            best_points.append(best_f)
        
        # Use remaining budget for random exploration
        used = num_main_samples + min(budget_per_sample, 5) * num_main_samples
        remaining = self.budget - used
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline - 3:
                    break
                x = lo + (hi - lo) * np.random.random(n)
                f = problem(x)
                best_points.append(f)
        
        F = np.array(best_points)
        
        return F
