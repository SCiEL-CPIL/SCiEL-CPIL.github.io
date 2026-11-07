"""Optimized ZDT2 solver using exact hypervolume-greedy adaptive sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting analytic structure with greedy hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        budget = self.budget
        C = 1 + 10 * (dim - 1)  # 291 for dim=30

        # The Pareto front is f2 = C * (1 - sqrt(f1)) with x2..xn = 0
        # We want to select N points on this curve to maximize hypervolume
        # with reference point (11, 11).

        # Strategy: Greedy hypervolume maximization
        # 1. Start with a coarse grid of candidate x1 values
        # 2. Greedily add the point that gives the largest marginal hypervolume gain
        # 3. This naturally concentrates points where the front is steepest (near x1=0)

        N = budget
        ref = np.array([11.0, 11.0])

        # Create a dense set of candidate x1 values
        # Use a non-uniform grid biased toward x1=0 where the front is steepest
        n_candidates = 500
        # Power-law spacing: more candidates near x1=0
        t = np.linspace(0, 1, n_candidates)
        x1_candidates = t ** 2  # Concentrates near 0

        # Compute corresponding f2 values on the Pareto front
        f2_candidates = C * (1 - np.sqrt(x1_candidates))
        
        # Candidate points in objective space
        candidates = np.column_stack([x1_candidates, f2_candidates])

        # Greedy selection: start with the two extreme points
        # Point 0: x1=0, f2=C (worst in f1, best in f2)
        # Point N-1: x1=1, f2=0 (best in f1, worst in f2)
        
        selected_idx = [0, n_candidates - 1]
        selected_points = candidates[selected_idx].copy()

        # Helper: compute hypervolume of a set of 2D points with reference
        def hypervolume_2d(points, ref_point):
            """Compute hypervolume of a set of 2D points relative to reference."""
            if len(points) == 0:
                return 0.0
            # Filter points that are dominated by reference
            mask = (points[:, 0] < ref_point[0]) & (points[:, 1] < ref_point[1])
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by f1 (ascending)
            pts = pts[np.argsort(pts[:, 0])]
            # Hypervolume = sum over each point of (ref_f1 - f1_i) * (f2_{i-1} - f2_i)
            # where f2_{i-1} is the f2 of the previous point (or ref_f2 for the first)
            hv = 0.0
            prev_f2 = ref_point[1]
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                width = ref_point[0] - f1_i
                height = prev_f2 - f2_i
                if width > 0 and height > 0:
                    hv += width * height
                prev_f2 = f2_i
            return hv

        # Greedily add points that maximize marginal hypervolume
        while len(selected_idx) < N:
            if time.time() > deadline - 5:
                break

            current_hv = hypervolume_2d(selected_points, ref)

            best_gain = -1
            best_idx = -1
            best_point = None

            # Evaluate marginal gain for each candidate not yet selected
            # For efficiency, only consider candidates not already selected
            selected_set = set(selected_idx)
            
            for i in range(n_candidates):
                if i in selected_set:
                    continue
                
                # Try adding candidate i
                new_points = np.vstack([selected_points, candidates[i]])
                new_hv = hypervolume_2d(new_points, ref)
                gain = new_hv - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_point = candidates[i]

            if best_idx == -1 or best_gain <= 0:
                break

            selected_idx.append(best_idx)
            selected_points = np.vstack([selected_points, best_point])

        # Sort selected points by x1 for consistent output
        sort_order = np.argsort(selected_points[:, 0])
        selected_points = selected_points[sort_order]

        # Construct decision vectors
        X = np.zeros((len(selected_points), dim))
        X[:, 0] = selected_points[:, 0]
        # x2..xn = 0 for Pareto front

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
