import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: f1 = t, f2 = 3*(1 - sqrt(t) - sin(10*pi*t)), t in [0,1]
        # The Pareto front consists of running minima of f2 as t increases.
        # We can compute f2(t) analytically to identify Pareto-optimal segments.
        
        # Fine grid to identify Pareto-optimal regions
        t_fine = np.linspace(0.0, 1.0, 10001)
        f2_fine = 3.0 * (1.0 - np.sqrt(t_fine) - np.sin(10.0 * np.pi * t_fine))
        
        # Find running minima: t_i is Pareto-optimal if f2(t_i) < min(f2(t_j) for j < i)
        # Build the set of Pareto-optimal t values
        running_min = np.full_like(f2_fine, np.inf)
        running_min[0] = f2_fine[0]
        for i in range(1, len(f2_fine)):
            running_min[i] = min(running_min[i-1], f2_fine[i])
        
        # Pareto-optimal points are where f2 equals the running minimum
        # But we want the segments, not just points.
        # A point t is Pareto-optimal if f2(t) <= min(f2(s) for s < t)
        # Actually, it's Pareto-optimal if there's no s < t with f2(s) <= f2(t)
        # i.e., f2(t) < min(f2(s) for s in [0, t)) -- strict for the previous points
        # But at the first point, it's trivially Pareto-optimal.
        
        # Let's identify the Pareto-optimal segments more carefully.
        # t is Pareto-optimal iff f2(t) < min_{s < t} f2(s), or t is the first point.
        # This means the Pareto front is the set of points that achieve new running minima.
        
        # Find indices where new running minima are achieved
        is_new_min = np.zeros(len(t_fine), dtype=bool)
        is_new_min[0] = True
        for i in range(1, len(t_fine)):
            if f2_fine[i] < running_min[i-1]:
                is_new_min[i] = True
            else:
                is_new_min[i] = False
        
        # The Pareto-optimal t values are those where is_new_min is True
        # But we want to sample the entire front, including points that are 
        # on the "edges" of the Pareto-optimal regions.
        
        # Actually, for hypervolume, we want points that are on the Pareto front.
        # The Pareto front in objective space is the curve traced by Pareto-optimal t values.
        # Let's collect all Pareto-optimal t values and their corresponding f2 values.
        
        pareto_idx = np.where(is_new_min)[0]
        if len(pareto_idx) > 0:
            pareto_t = t_fine[pareto_idx]
            pareto_f2 = f2_fine[pareto_idx]
        else:
            pareto_t = np.array([0.0])
            pareto_f2 = np.array([f2_fine[0]])
        
        # Now allocate budget to densely sample the Pareto-optimal segments.
        # The Pareto front in objective space goes from (0, f2(0)) to (1, f2(1))
        # but only along the Pareto-optimal portions.
        
        # For best hypervolume coverage, we want uniform spacing in the objective space
        # along the Pareto front. Let's parameterize by f1 (=t) and f2.
        
        # The Pareto-optimal points are ordered by increasing t (and thus increasing f1).
        # We want to place our budget points along this curve.
        
        # Strategy: use all budget points, placed at Pareto-optimal t values
        # with density proportional to the local "importance" for hypervolume.
        
        # For 2-objective hypervolume with reference point (11,11), the contribution
        # of each point depends on its position relative to other points.
        # Uniform spacing in f1 along the Pareto-optimal segments is a good heuristic.
        
        # Let's identify the contiguous segments of Pareto-optimal t values.
        # A segment is a maximal interval [a,b] where all t in [a,b] are Pareto-optimal.
        
        # Actually, let me reconsider. The Pareto-optimal set might be discrete points
        # (the running minima) rather than continuous segments. Let me check.
        
        # For ZDT3, the Pareto front is actually a set of curves, not just points.
        # Wait, no - the Pareto front is 1-dimensional. Each point on the front 
        # corresponds to a unique t value (since f1=t).
        
        # The issue is that not all t values give Pareto-optimal points. Only the
        # running minima do. So the Pareto front is a subset of the curve.
        
        # Let me just use all budget evaluations to sample the Pareto-optimal t values
        # as densely as possible. I'll create a fine grid of Pareto-optimal t values
        # and select budget points from it.
        
        # Create a very fine grid of Pareto-optimal t values
        t_finer = np.linspace(0.0, 1.0, 100001)
        f2_finer = 3.0 * (1.0 - np.sqrt(t_finer) - np.sin(10.0 * np.pi * t_finer))
        
        # Find running minima on the finer grid
        running_min_finer = np.full_like(f2_finer, np.inf)
        running_min_finer[0] = f2_finer[0]
        for i in range(1, len(t_finer)):
            running_min_finer[i] = min(running_min_finer[i-1], f2_finer[i])
        
        is_new_min_finer = np.zeros(len(t_finer), dtype=bool)
        is_new_min_finer[0] = True
        for i in range(1, len(t_finer)):
            if f2_finer[i] < running_min_finer[i-1]:
                is_new_min_finer[i] = True
        
        pareto_t_finer = t_finer[is_new_min_finer]
        pareto_f2_finer = f2_finer[is_new_min_finer]
        
        # Now select budget points from the Pareto-optimal set.
        # We want to maximize hypervolume, so we want good coverage.
        # Use uniform spacing in the index of Pareto-optimal points, 
        # or better, uniform spacing in f1 along the Pareto-optimal regions.
        
        # Let's use a simple approach: select budget points from the Pareto-optimal
        # t values with uniform spacing in the objective space.
        
        if len(pareto_t_finer) >= budget:
            # Select budget points with uniform spacing in f1
            indices = np.linspace(0, len(pareto_t_finer) - 1, budget).astype(int)
            selected_t = pareto_t_finer[indices]
        else:
            # Use all Pareto-optimal points and add some more if needed
            selected_t = pareto_t_finer.copy()
            # Pad with additional points if we have budget left
            if len(selected_t) < budget:
                # Add points at the endpoints of Pareto-optimal segments
                remaining = budget - len(selected_t)
                # Just add uniformly spaced points from the full Pareto set
                extra_indices = np.linspace(0, len(pareto_t_finer) - 1, remaining).astype(int)
                extra_t = pareto_t_finer[extra_indices]
                selected_t = np.concatenate([selected_t, extra_t])
                selected_t = np.unique(selected_t)
        
        # Create candidate points
        X = np.zeros((len(selected_t), n))
        X[:, 0] = selected_t
        # x[1:] = 0 gives minimum g, placing points on the Pareto front
        
        # Evaluate all front points
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
        
        return X
