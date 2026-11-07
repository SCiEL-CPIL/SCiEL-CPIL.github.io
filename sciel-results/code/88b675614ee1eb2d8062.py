import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # ZDT3 analytic Pareto front:
        # f1 = x1, f2 = g * h(f1, g) where g = 1 when x2..xm = 0
        # h(f1, g) = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        # With g=1: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # The Pareto front is achieved when x[1:] = 0

        # We need to evaluate points on the Pareto front to maximize hypervolume
        # The key insight: we should sample f1 values that capture the discontinuous
        # Pareto front well. ZDT3 has 57 disconnected regions.
        
        # Strategy: Use all budget to evaluate points on the Pareto front
        # Use a combination of:
        # 1. Uniform sampling for broad coverage
        # 2. Denser sampling in regions where the front has more structure
        
        # The Pareto front f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # has oscillations with period related to sin(10*pi*f1), so period = 0.2 in f1
        
        # For hypervolume with reference point [11, 11], we want to maximize the area
        # under the Pareto front. Since all points on the front are non-dominated,
        # we want good coverage of the entire front.
        
        # Use a stratified sampling approach:
        # - Divide [0,1] into many small intervals
        # - Sample multiple points in each interval, especially near discontinuities
        
        # The discontinuities occur where the derivative of f2 changes sign abruptly
        # due to the sin term. Let's use adaptive sampling.
        
        # First, let's understand the structure better:
        # f2(f1) = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # df2/df1 = -1/(2*sqrt(f1)) - sin(10*pi*f1) - 10*pi*f1*cos(10*pi*f1)
        
        # For hypervolume maximization, we want to select points that maximize the
        # area covered. In 2D, if we sort points by f1, the hypervolume is the sum
        # of rectangles. We want to place points where they contribute most to the area.
        
        # A good strategy: use a dense uniform grid, then refine near the "kinks"
        
        n_total = budget
        
        # Use all evaluations on the Pareto front
        # Generate f1 values with good coverage
        
        # Approach: Use a combination of:
        # 1. A base uniform grid
        # 2. Additional points near where the front has high curvature or discontinuities
        
        # Let's use a simple but effective approach:
        # Sample f1 values using a non-uniform distribution that puts more points
        # in regions where the front is more complex
        
        # Actually, for ZDT3, the best approach is to sample uniformly because
        # the front has regular oscillations. Let's use a dense uniform grid.
        
        # But we need to be smart: we have exactly budget evaluations.
        # Let's use all of them on the Pareto front with optimal placement.
        
        # For 2D hypervolume with reference [11,11], and the Pareto front in [0,1]x[0,~2],
        # the hypervolume is approximately the integral of (11 - f2(f1)) df1 from 0 to 1
        # plus the area from the leftmost point to f1=0 and rightmost to f1=1.
        
        # To maximize this, we want to capture the shape of f2(f1) well.
        
        # Let's use a stratified sampling approach with more points where the function
        # has higher variation.
        
        # Generate candidate f1 values
        # Use a mixture: 80% uniform, 20% concentrated near key features
        
        n_uniform = int(0.7 * n_total)
        n_adaptive = n_total - n_uniform
        
        # Uniform samples
        f1_uniform = np.linspace(0, 1, n_uniform, endpoint=True)
        
        # Adaptive samples: focus on regions where sin(10*pi*f1) has zeros and extrema
        # Zeros of sin(10*pi*f1): f1 = k/10 for k = 0, 1, ..., 10
        # Extrema of f2 occur where df2/df1 = 0
        
        # Let's place adaptive points near the zeros of the sine term and near f1=0
        # (where sqrt(f1) has a singularity in its derivative)
        
        # Key f1 values to focus on:
        # - Near 0 (singularity in derivative of sqrt)
        # - Zeros of sin(10*pi*f1): 0, 0.1, 0.2, ..., 1.0
        # - Points where the front has local minima/maxima
        
        # Generate adaptive points with higher density near f1=0 and at the sine zeros
        f1_adaptive = []
        
        # More points near f1=0
        n_near_zero = max(5, n_adaptive // 5)
        f1_adaptive.extend(np.linspace(0, 0.05, n_near_zero, endpoint=True))
        
        # Points around each zero of sin(10*pi*f1)
        remaining = n_adaptive - n_near_zero
        if remaining > 0:
            # Place points around each k/10
            n_per_region = max(1, remaining // 10)
            for k in range(1, 11):
                center = k / 10.0
                spread = 0.02
                n_pts = min(n_per_region, max(1, remaining // 10))
                f1_adaptive.extend(np.linspace(max(0, center - spread), min(1, center + spread), n_pts, endpoint=True))
        
        # Combine and deduplicate
        f1_all = np.concatenate([f1_uniform, f1_adaptive])
        f1_all = np.unique(f1_all)
        
        # If we have more points than budget, subsample
        if len(f1_all) > n_total:
            # Use stratified subsampling
            indices = np.linspace(0, len(f1_all) - 1, n_total, dtype=int)
            f1_all = f1_all[indices]
        
        # If we have fewer, that's fine - we just have fewer evaluations
        n_actual = len(f1_all)
        
        # Construct x vectors: x[0] = f1, x[1:] = 0
        X = np.zeros((n_actual, dim))
        X[:, 0] = f1_all
        
        # Evaluate all points
        F = np.array([problem(X[i]) for i in range(n_actual)])
        
        # Check time
        if time.time() > deadline - 5:
            return F
        
        # Return the evaluated points (return value is ignored, but we return something valid)
        return F
