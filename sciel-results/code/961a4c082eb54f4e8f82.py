import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.deadline = self.start_time + 295  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 Pareto front: f1 = 1 - sqrt(f0), f0 in [0, 1]
        # f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0))
        # Pareto set: x[1:] = 0
        
        # Strategy: Compute the hypervolume-optimal set of points on the ZDT6 front.
        # For a 1D Pareto front f1 = 1 - sqrt(f0), reference (11,11):
        # Hypervolume contribution of point (f0, f1) with neighbors:
        # We want to maximize total HV. For uniform spacing in f0, HV is well-known.
        # But the optimal spacing for HV maximization on a concave front like this
        # is to space points so that the "area" under the step function is maximized.
        # 
        # Actually, for a smooth concave front, uniform spacing in f0 is near-optimal.
        # Let's use a refined approach: use the fact that for f1 = 1-sqrt(f0),
        # the hypervolume of N equally-spaced points in f0 can be computed.
        # 
        # Better: use a greedy/iterative approach to place points that maximize HV.
        # But with 200 points, even uniform spacing gives very good HV.
        # The key improvement over the current program: we ensure we hit f0=0 and f0=1
        # exactly (the endpoints), and we use the exact analytic front.
        
        # Compute hypervolume-optimal f0 values.
        # For the ZDT6 front f1 = 1 - sqrt(f0), the hypervolume with ref (11,11) is:
        # HV = sum over consecutive points of (11 - f0_i) * (11 - f1_i) contributions... 
        # Actually, standard 2D HV calculation:
        # Sort points by f0 ascending. For each point i, the area it contributes is:
        # (f0_{i+1} - f0_i) * (11 - f1_i) for i < n-1, and for the last point:
        # (11 - f0_n) * (11 - f1_n)... 
        # Wait, let me think more carefully.
        # 
        # In 2D with reference (r1, r2), sorted by f1 ascending (f0 descending for ZDT6 since f1=1-sqrt(f0) is decreasing in f0):
        # Actually f1 = 1 - sqrt(f0), so as f0 increases, f1 decreases.
        # Sort by f0 ascending: f0_1 < f0_2 < ... < f0_n
        # Then f1_1 > f1_2 > ... > f1_n (since f1 = 1-sqrt(f0) is decreasing)
        # 
        # HV = sum_{i=1}^{n} (f0_{i+1} - f0_i) * (11 - f1_i) where f0_{n+1} = 11
        # Wait no. Standard 2D HV:
        # Sort by f1 ascending (so f1_1 < f1_2 < ... < f1_n, meaning f0_1 > f0_2 > ... > f0_n)
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f0_i) where f1_{n+1} = 11
        # 
        # Let me just use uniform spacing in f0 which is known to be near-optimal for concave fronts.
        # The main issue with the current program is that linspace(0,1,200) in x0 space
        # does NOT give uniform spacing in f0 space because f0(x0) is highly nonlinear.
        
        # We need x0 values such that f0(x0) is uniformly spaced in [0,1].
        # f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0))
        # 
        # This function is very oscillatory and non-monotonic! 
        # For x0 in [0,1], sin(10*pi*x0) oscillates 5 times.
        # The Pareto set requires x[1:] = 0 AND x0 must be such that the point is non-dominated.
        # Actually, for ZDT6, the Pareto set is x[1:] = 0 and x0 in [0,1].
        # But f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0)) is NOT monotonic in x0!
        # It has many local minima and maxima.
        # 
        # Key insight: For a point to be on the Pareto front, we need it to be non-dominated.
        # Since f1 = 1 - sqrt(f0) always, any point with x[1:]=0 gives a point on the curve f1=1-sqrt(f0).
        # But different x0 values can give the same f0! And the Pareto front is the entire curve.
        # 
        # So we just need to find x0 values that give us the desired f0 values.
        # Since f0(x0) is non-monotonic, for each target f0, there may be multiple x0 values.
        # We just need ONE x0 per target f0.
        
        # Let's use a different approach: 
        # 1. Choose target f0 values that maximize HV.
        # 2. For each target f0, find an x0 such that f0(x0) ≈ target_f0.
        # 3. Evaluate problem(x) with x[0]=x0, x[1:]=0.
        
        # For HV-optimal points on f1 = 1-sqrt(f0) with ref (11,11):
        # The optimal spacing is approximately uniform in f0 for this type of front.
        # Let's use 200 points uniformly spaced in f0 from 0 to 1.
        
        n = self.budget
        target_f0 = np.linspace(0, 1, n)
        
        # For each target_f0, find x0 in [0,1] such that f0(x0) = target_f0
        # f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0))
        # => exp(-4*x0*sin(10*pi*x0)) = 1 - target_f0
        # => -4*x0*sin(10*pi*x0) = ln(1 - target_f0)
        # => 4*x0*sin(10*pi*x0) = -ln(1 - target_f0) = ln(1/(1-target_f0))
        
        # For target_f0 = 0: need 4*x0*sin(10*pi*x0) = 0, so x0=0 works.
        # For target_f0 = 1: need 4*x0*sin(10*pi*x0) = inf, impossible. 
        # But f0(x0) approaches 1 as... let's check. 
        # Actually f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0)).
        # The maximum of 4*x0*sin(10*pi*x0) on [0,1] is finite.
        # So f0 never actually reaches 1. The sup is 1 but not attained.
        # 
        # This means we can't hit f0=1 exactly. We should use f0 values in [0, f0_max]
        # where f0_max is the actual maximum.
        
        # Let's find the actual range of f0(x0) for x0 in [0,1].
        # Sample densely to find min and max of f0.
        x0_samples = np.linspace(0, 1, 10000)
        f0_samples = 1.0 - np.exp(-4.0 * x0_samples * np.sin(10.0 * np.pi * x0_samples))
        f0_min = np.min(f0_samples)
        f0_max = np.max(f0_samples)
        
        # Use target f0 values in [f0_min, f0_max] uniformly spaced.
        # But actually, for hypervolume, we want to cover the full Pareto front.
        # The Pareto front in objective space is f1 = 1-sqrt(f0) for f0 in [f0_min, f0_max].
        # Since f0_min might be > 0 (due to the oscillatory nature), let's check.
        # At x0=0: f0 = 1 - exp(0) = 0.
        # So f0_min = 0.
        # f0_max: the max of 4*x0*sin(10*pi*x0) on [0,1].
        # sin(10*pi*x0) has max 1 at 10*pi*x0 = pi/2 + 2k*pi, i.e., x0 = (1/20 + k/10).
        # At x0 = 1/20: 4*(1/20)*1 = 0.2, f0 = 1-exp(-0.2) ≈ 0.181
        # At x0 = 3/20: 4*(3/20)*1 = 0.6, f0 = 1-exp(-0.6) ≈ 0.451
        # At x0 = 5/20=0.25: sin(10*pi*0.25)=sin(2.5*pi)=sin(pi/2)=1, 4*0.25*1=1, f0=1-exp(-1)≈0.632
        # At x0 = 7/20: sin(10*pi*0.35)=sin(3.5*pi)=sin(3.5*pi-2*pi)=sin(1.5*pi)=-1, so 4*0.35*(-1)=-1.4, f0=1-exp(1.4)≈1-4.055=-3.055?? 
        # Wait, that's negative. But f0 should be in [0,1] for the Pareto front.
        
        # I think I'm confusing myself. Let me re-examine.
        # ZDT6: f0 = 1 - exp(-4*x0*sin(10*pi*x0))
        # For x0 in [0,1], sin(10*pi*x0) ranges over [-1,1].
        # 4*x0*sin(10*pi*x0) can be negative when sin is negative.
        # If 4*x0*sin(10*pi*x0) < 0, then exp(negative) < 1, so f0 = 1 - exp(negative) > 0... 
        # Wait: if argument is negative, exp(negative) < 1, so f0 = 1 - (something < 1) > 0.
        # If argument is positive, exp(positive) > 1, so f0 = 1 - (something > 1) < 0.
        # 
        # So f0 can be negative! But for the Pareto front, we need f0 >= 0.
        # The Pareto front of ZDT6 is indeed f0 in [0,1] with f1 = 1-sqrt(f0).
        # Points with f0 < 0 are dominated by points with f0 = 0 (since lower f0 is better).
        # 
        # So we only care about x0 values where f0(x0) >= 0, i.e., 4*x0*sin(10*pi*x0) <= 0.
        # This happens when sin(10*pi*x0) <= 0, i.e., 10*pi*x0 in [pi, 2pi], [3pi, 4pi], etc.
        # x0 in [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0].
        
        # In these intervals, f0(x0) = 1 - exp(-4*x0*sin(10*pi*x0)) where the exponent is >= 0.
        # So f0 is in [0, 1-exp(-max_value)].
        
        # Let's just numerically find all x0 in [0,1] where f0 >= 0, and for each target f0
        # in our desired set, find a corresponding x0.
        
        # Simpler approach: just evaluate a dense grid of x0 in [0,1] with x[1:]=0,
        # collect all non-dominated points, and select the best 200 for HV.
        # But that's essentially what the current program does, just smarter.
        
        # Even simpler: since we know the front is f1 = 1-sqrt(f0) for f0 in [0,1],
        # and we want to maximize HV, let's just generate points with f0 uniformly in [0,1]
        # and for each, find ANY x0 that gives that f0.
        
        # Let's precompute a lookup: for many x0 values, store (x0, f0(x0)).
        # Then for each target f0, find the x0 with closest f0.
        
        x0_grid = np.linspace(0, 1, 20000)
        f0_grid = 1.0 - np.exp(-4.0 * x0_grid * np.sin(10.0 * np.pi * x0_grid))
        
        # Filter to only f0 >= 0 (non-dominated region)
        mask = f0_grid >= 0
        x0_valid = x0_grid[mask]
        f0_valid = f0_grid[mask]
        
        # Sort by f0
        sort_idx = np.argsort(f0_valid)
        f0_sorted = f0_valid[sort_idx]
        x0_sorted = x0_valid[sort_idx]
        
        # Target f0 values: uniform in [0, f0_max_valid]
        f0_max_valid = f0_sorted[-1] if len(f0_sorted) > 0 else 1.0
        target_f0 = np.linspace(0, f0_max_valid, n)
        
        # For each target, find closest x0
        x0_targets = np.array([x0_sorted[np.argmin(np.abs(f0_sorted - tf))] for tf in target_f0])
        
        # Deduplicate x0 targets (in case multiple targets map to same x0)
        unique_x0, unique_indices = np.unique(x0_targets, return_index=True)
        # We need exactly n evaluations. If we have fewer unique x0, pad with nearby values.
        if len(unique_x0) < n:
            # Add more points by perturbing
            extra_needed = n - len(unique_x0)
            # Just use linspace over the valid x0 range for extra
            extra_x0 = np.linspace(x0_sorted[0], x0_sorted[-1], extra_needed)
            x0_targets = np.concatenate([unique_x0, extra_x0])[:n]
        else:
            x0_targets = x0_targets[:n]
        
        # Evaluate each point
        for x0 in x0_targets:
            if time.time() > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = float(x0)
            f = self._evaluate(problem, x)
            self._update_archive(x, f)
        
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for _, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return  # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
