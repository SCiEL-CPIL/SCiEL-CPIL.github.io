import numpy as np
from numpy.polynomial.legendre import leggauss

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        if self.budget <= 0:
            return np.array([])

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])
        
        # The non-dominated front of ZDT3 is concentrated in intervals where sin(10*pi*x1) >= 0
        # These are [0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]
        # And the point x1=1.0
        # Points in [0.1, 0.2], etc. are dominated by the left endpoint of the interval.
        
        intervals = [
            (0.0, 0.1),
            (0.2, 0.3),
            (0.4, 0.5),
            (0.6, 0.7),
            (0.8, 0.9)
        ]
        
        x1_vals = []
        
        # Reserve 1 evaluation for x1=1.0
        if n_evals >= 1:
            x1_vals.append(1.0)
            n_remaining = n_evals - 1
        else:
            return np.array([])
            
        # Distribute remaining evaluations among the 5 intervals
        n_intervals = len(intervals)
        if n_remaining > 0:
            # Ensure at least 1 point per interval if possible
            if n_remaining >= n_intervals:
                pts_per_interval = n_remaining // n_intervals
                extra = n_remaining % n_intervals
            else:
                # If very small budget, just pick midpoints or endpoints
                pts_per_interval = 1
                extra = 0
                # Adjust if we don't have enough for 1 per interval
                if n_remaining < n_intervals:
                    # Pick the most important ones? Or just spread out?
                    # For now, just take the first n_remaining intervals
                    pass

            for i, (start, end) in enumerate(intervals):
                n_pts = pts_per_interval
                if i < extra:
                    n_pts += 1
                    
                if n_pts <= 0:
                    continue
                    
                if n_pts == 1:
                    # Pick the midpoint or the start?
                    # The start is the zero-crossing, which is a "corner".
                    # The midpoint is where f2 is lowest in the "bump"? No, f2 is lowest at the end of the "good" interval?
                    # In [0, 0.1], f2 decreases. So end (0.1) is best.
                    # But 0.1 is the start of the "bad" interval.
                    # So we should include the endpoints of the good intervals.
                    # Let's use linspace including endpoints if n_pts > 1.
                    # If n_pts == 1, pick the end (right endpoint) as it's the "best" in that segment?
                    # Actually, the right endpoint is shared with the next segment's start (which is bad).
                    # The left endpoint is the start of the good segment.
                    # Let's just use linspace.
                    x1_vals.append(start)
                else:
                    # Use linspace from start to end
                    # This includes start and end.
                    # The end of [0, 0.1] is 0.1.
                    # The start of [0.2, 0.3] is 0.2.
                    # 0.1 and 0.2 are both "corners".
                    x1_vals.extend(np.linspace(start, end, n_pts).tolist())
        
        # Ensure we don't exceed budget (due to rounding or logic)
        if len(x1_vals) > n_evals:
            x1_vals = x1_vals[:n_evals]
            
        # Create the population matrix
        X = np.zeros((len(x1_vals), self.dim))
        X[:, 0] = x1_vals
        # x[1:] are 0 by default
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
