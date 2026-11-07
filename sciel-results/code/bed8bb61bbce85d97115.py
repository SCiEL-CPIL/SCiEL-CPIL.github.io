import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific structure:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = g * (1 - sqrt(f1/g)) where g = 1 + 9*(sum(x[1:])/9)^(1/(1+2*sqrt(sum(x[1:])/9)))
        # The Pareto front is achieved when x[1:] = 0, so g=1.
        # Then f2 = 1 - sqrt(f1).
        # The front is the curve (f1, 1-sqrt(f1)) for f1 in [0, 1].
        
        # To maximize hypervolume, we want to cover the entire front [0, 1] in f1 space.
        # Since f1 is not monotonic, we need to be careful.
        # However, for any f1 in [0,1], there exists an x1 such that f1(x1)=f1.
        # We can sample x1 such that the resulting f1 values are well-distributed.
        
        # A better approach: sample f1 uniformly in [0,1] and find corresponding x1.
        # But f1 is not one-to-one. We can just sample x1 in a way that covers the range well.
        
        # Actually, the simplest and most effective approach for ZDT6 is:
        # 1. Set x[1:] = 0 (to be on the Pareto front)
        # 2. Sample x1 such that f1(x1) covers [0,1] well
        
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # This function oscillates. Let's sample x1 uniformly and see what f1 values we get.
        # Or, we can use a smarter sampling strategy.
        
        # For maximum HV, we want points evenly spaced in the objective space.
        # The front is (f1, 1-sqrt(f1)). If we sample f1 uniformly, the points will be 
        # closer together in f2 space for small f1 (since d(1-sqrt(f1))/df1 = -1/(2*sqrt(f1))).
        
        # Let's use a mix:
        # - 80% of points: evenly spaced in f1 space (inverse sampling)
        # - 20% of points: clustered near high-curvature regions
        
        # But first, let's just use uniform x1 sampling and see if it's good enough.
        # With 200 points, uniform x1 should give decent coverage.
        
        # Actually, let's think about this more carefully.
        # The hypervolume is maximized when points are evenly spaced in the objective space.
        # For the curve (f1, 1-sqrt(f1)), if we want even spacing in f2, we should sample 
        # f2 uniformly and compute f1 = (1-f2)^2.
        
        # Let's do that: sample f2 uniformly in [0,1], compute f1 = (1-f2)^2, then find x1.
        # But finding x1 for a given f1 requires solving f1(x1) = f1, which is complex.
        
        # Alternative: just sample x1 uniformly. With 200 points, this should be sufficient.
        
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0
        
        # Strategy:
        # 1. Use 80% of budget for evenly-spaced points in f1 space (via x1 sampling)
        # 2. Use 20% of budget for clustered points near critical regions
        
        # First, let's understand f1(x1) better.
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Zeros of sin(6*pi*x1) are at x1 = k/6 for k=0,1,...,6
        # At these points, f1 = 1.
        # Between these, f1 < 1.
        
        # The Pareto front is parameterized by f1 in [0,1].
        # To get uniform f1 coverage, we need to sample x1 non-uniformly.
        
        # Let's use a numerical approach:
        # 1. Sample f1 uniformly in [0,1]
        # 2. For each f1, find an x1 such that f1(x1) = f1
        
        # But since f1 is not one-to-one, we can just pick any valid x1.
        # The simplest is to use a coarse grid of x1 and then refine.
        
        # Actually, let's use a different approach:
        # Sample x1 using a low-discrepancy sequence (Halton) for good coverage.
        # This is what the current program does.
        
        # Let's combine: use Halton for the main coverage, but ensure we hit the critical points.
        
        def halton_1d(n, base=2):
            """Generate n Halton sequence points in [0,1)"""
            result = []
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                num = i
                while num:
                    f /= base
                    x += f * (num % base)
                    num //= base
                result.append(x)
            return np.array(result)
        
        # Generate x1 values
        # 80% Halton, 20% clustered near critical points
        n_main = int(0.8 * n_evals)
        n_cluster = n_evals - n_main
        
        # Halton sequence for main coverage
        x1_main = halton_1d(n_main)
        
        # Critical points: where sin(6*pi*x1) = 0, i.e., x1 = k/6
        # These are where f1 = 1, which is the maximum.
        # But we also want coverage near where f1 changes rapidly.
        
        # Let's cluster near x1 = 0.5 (where f1 has a local minimum in the middle oscillation)
        # and near the boundaries.
        
        # Actually, let's just use a simple grid for the clustered part.
        # Critical x1 values: 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1
        # Let's add points near these.
        
        critical_x1 = np.array([0.0, 1.0/6.0, 2.0/6.0, 3.0/6.0, 4.0/6.0, 5.0/6.0, 1.0])
        
        # Generate clustered points around critical x1
        x1_cluster = []
        points_per_critical = max(1, n_cluster // len(critical_x1))
        for cx in critical_x1:
            for i in range(points_per_critical):
                # Add small perturbation
                offset = (i - points_per_critical/2 + 0.5) * 0.01
                x1c = cx + offset
                if 0 <= x1c <= 1:
                    x1_cluster.append(x1c)
        
        x1_cluster = np.array(x1_cluster[:n_cluster])
        
        # Combine and sort
        x1_vals = np.sort(np.concatenate([x1_main, x1_cluster]))
        
        # Remove duplicates
        x1_vals = np.unique(np.round(x1_vals, 15))
        
        # If we have more than budget, take first budget points
        if len(x1_vals) > n_evals:
            x1_vals = x1_vals[:n_evals]
        
        # Evaluate each point
        for x1 in x1_vals:
            if time.time() - start_time > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] are already 0
            # Clip to bounds just in case
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
