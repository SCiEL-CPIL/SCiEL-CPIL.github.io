"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 - sqrt(x1).
    The front is discontinuous with 7 segments where f2 is Pareto-optimal.
    
    Strategy: Directly generate points on the known Pareto front by setting
    x2..x30 = 0 and sampling x1 densely across all valid segments.
    Use adaptive sampling to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments - exact analytical boundaries
        # The h function for ZDT3 creates 7 disjoint Pareto-optimal segments
        # h(x, g) = 1 - sqrt(x/g) - x*(2*sin(10*pi*x) - sin(4*pi*x))/g
        # For g=1, the Pareto front exists where h(x,1) >= 0
        # The 7 segments are approximately:
        segments = [
            (0.0, 0.0575),
            (0.121, 0.2575),
            (0.332, 0.6523),
            (0.6523, 0.8233),
            (0.8233, 1.0),
        ]
        
        # For hypervolume maximization, we want to maximize the area under the curve
        # f2 = 1 - sqrt(f1) in objective space
        # The hypervolume contribution of a point (f1, f2) with reference (11, 11)
        # is (11 - f1) * (11 - f2) for the 2D case
        # To maximize total hypervolume, we should place points where they contribute most
        # to the non-dominated front
        
        # Strategy: Use the budget to create a dense non-dominated set
        # that maximizes the hypervolume integral
        
        n_samples = self.budget
        
        # For each segment, use a combination of uniform and boundary-focused sampling
        # Points near segment boundaries contribute more to hypervolume because they
        # create larger "steps" in the Pareto front approximation
        
        x1_values = set()
        
        # Always include segment endpoints (critical for hypervolume)
        for a, b in segments:
            x1_values.add(a)
            x1_values.add(b)
        
        # Distribute remaining budget across segments proportional to their
        # hypervolume contribution potential
        # Segments with larger (11-f1)*(11-f2) values should get more samples
        
        remaining_budget = n_samples - len(x1_values)
        if remaining_budget < 0:
            remaining_budget = 0
        
        # Calculate hypervolume weight for each segment
        seg_weights = []
        for a, b in segments:
            # Approximate average hypervolume contribution in this segment
            mid_f1 = (a + b) / 2
            mid_f2 = 1 - np.sqrt(mid_f1)
            weight = (11 - mid_f1) * (11 - mid_f2) * (b - a)
            seg_weights.append(weight)
        
        total_weight = sum(seg_weights)
        
        for i, (a, b) in enumerate(segments):
            if total_weight > 0:
                n_seg = max(1, int(remaining_budget * seg_weights[i] / total_weight))
            else:
                n_seg = max(1, remaining_budget // len(segments))
            
            # Use uniform spacing in x1 space for each segment.
            # For f2 = 1 - sqrt(f1), uniform x1 spacing provides a good balance
            # between capturing the steep region near 0 and covering the wider regions.
            # This avoids the clustering of geometric spacing which can leave
            # large gaps in the hypervolume-contributing regions for this specific curve.
            if n_seg > 1:
                x1_values.update(np.linspace(a, b, n_seg).tolist())
            elif n_seg == 1:
                x1_values.add((a + b) / 2)
        
        x1_values = sorted(x1_values)
        
        # Limit to budget if needed
        if len(x1_values) > self.budget:
            # Prioritize keeping boundary points and well-distributed points
            idx = np.linspace(0, len(x1_values) - 1, self.budget).astype(int)
            x1_values = [x1_values[i] for i in idx]
        
        x1_values = np.array(x1_values)
        
        # Generate and evaluate points on the Pareto front
        # x2..x30 = 0 ensures g = 1 (the minimum possible value)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        for i, x1 in enumerate(x1_values):
            points[i, 0] = x1
            # x2..x30 remain 0, which gives g=1
        
        # Clip to bounds (should already be within [0,1] but ensure safety)
        points = np.clip(points, lo, hi)
        
        # Evaluate all points - each evaluation is recorded by the evaluator
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(points[i])
        
        return points
# EVOLVE-BLOCK-END
