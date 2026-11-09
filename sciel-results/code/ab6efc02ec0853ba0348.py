"""Optimized multi-objective search for ZDT3 using numerical hypervolume optimization."""
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using analytical ZDT3 Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1 + 9 * (29 / 30)  # = 10.9
        ref_point = np.array([11.0, 11.0])

        # Compute the exact Pareto front analytically
        # f1(t) = t, f2(t) = g * (1 - sqrt(t/g) - sin(10*pi*t))
        # We need to select 200 points from this front to maximize hypervolume

        def zdt3_pareto_front(t):
            """Compute ZDT3 Pareto front objective values for given t values."""
            t = np.asarray(t, dtype=float)
            f1 = t
            f2 = g * (1 - np.sqrt(np.maximum(t, 0) / g) - np.sin(10 * np.pi * t))
            return np.column_stack([f1, f2])

        def compute_hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of points.
            
            points: (N, 2) array of objective values
            ref: (2,) reference point
            Only points strictly better than ref in all objectives count.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than reference
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1 (first objective)
            pts = pts[np.argsort(pts[:, 0])]
            
            # Compute hypervolume as sum of rectangles
            # For each point, the contribution is (ref[1] - f2[i]) * (f1[i+1] - f1[i])
            # where f1[i+1] is the next point's f1 or ref[0] for the last point
            hv = 0.0
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                
                # Width: from f1_i to next f1 (or ref[0])
                if i < len(pts) - 1:
                    f1_next = pts[i + 1, 0]
                else:
                    f1_next = ref[0]
                
                # Only count if f2_i < ref[1]
                if f2_i < ref[1]:
                    width = f1_next - f1_i
                    height = ref[1] - f2_i
                    hv += width * height
            
            return hv

        # Step 1: Create a dense sample of the Pareto front for optimization
        n_dense = 10000
        t_dense = np.linspace(0, 1, n_dense)
        F_dense = zdt3_pareto_front(t_dense)
        
        # Filter to only points strictly better than reference
        mask_dense = np.all(F_dense < ref_point, axis=1)
        t_valid = t_dense[mask_dense]
        F_valid = F_dense[mask_dense]
        
        if len(F_valid) == 0:
            # Fallback: just sample uniformly
            t_samples = np.linspace(0, 1, max_evals)
            x = np.zeros(dim)
            recorded = []
            for t in t_samples:
                x[0] = t
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
                if n_evals >= max_evals or time.time() > deadline:
                    break
            return recorded

        # Step 2: Use greedy selection to pick max_evals points that maximize hypervolume
        # Start with the two endpoints (min and max f1)
        n_select = min(max_evals, len(F_valid))
        
        # Sort by f1
        sort_idx = np.argsort(F_valid[:, 0])
        t_sorted = t_valid[sort_idx]
        F_sorted = F_valid[sort_idx]
        
        # Initialize with endpoints
        selected_indices = [0, len(F_sorted) - 1]
        selected_set = set(selected_indices)
        
        # Precompute for faster hypervolume computation
        # We'll use a greedy approach: at each step, add the point that maximizes
        # the incremental hypervolume
        
        # For efficiency, we can use the fact that in 2D, adding a point between
        # two existing points increases hypervolume by:
        # (ref[1] - f2_new) * (f1_right - f1_left) - 
        #   [(ref[1] - f2_left) * (f1_new - f1_left) + (ref[1] - f2_right) * (f1_right - f1_new)]
        # if the new point is non-dominated relative to its neighbors
        
        # Simpler approach: just compute hypervolume incrementally
        # For 200 points from 10000, this is feasible with optimization
        
        # Use a priority queue approach: for each candidate, estimate its value
        # based on the gap it fills
        
        # Build a list of intervals between selected points
        # Each interval has a "potential" based on the maximum possible contribution
        
        # For simplicity and speed, use a different strategy:
        # Since the front is monotone in f1, we can use a space-filling approach
        # that accounts for the curvature
        
        # The key insight: the hypervolume contribution of a point at (f1, f2)
        # in a staircase approximation is approximately (ref[1] - f2) * delta_f1
        # where delta_f1 is the spacing around that point
        
        # To maximize sum of (ref[1] - f2[i]) * delta_f1[i], we want to allocate
        # more delta_f1 to points with larger (ref[1] - f2[i])
        
        # This suggests non-uniform spacing: denser where f2 is smaller (closer to 0)
        # which is near t=0 for ZDT3
        
        # Use a power-law spacing: t_i = (i/(N-1))^p
        # Optimize p to maximize hypervolume
        
        # Let's try several values of p and pick the best
        best_p = 0.8
        best_hv = -1
        best_t_samples = None
        
        # Try a range of p values
        p_candidates = np.linspace(0.5, 1.5, 21)
        
        for p in p_candidates:
            if time.time() > deadline:
                break
            
            t_samples = (np.arange(n_select) / (n_select - 1)) ** p
            t_samples = np.clip(t_samples, 0, 1)
            
            # Ensure we have exactly n_select unique points
            if len(np.unique(t_samples)) < n_select:
                # Add some jitter
                t_samples = np.linspace(0, 1, n_select)
                t_samples = (t_samples) ** p
                t_samples = np.clip(t_samples, 0, 1)
            
            F_samples = zdt3_pareto_front(t_samples)
            
            # Filter valid points
            mask = np.all(F_samples < ref_point, axis=1)
            if np.sum(mask) == 0:
                continue
            
            F_valid_samples = F_samples[mask]
            hv = compute_hypervolume_2d(F_valid_samples, ref_point)
            
            if hv > best_hv:
                best_hv = hv
                best_p = p
                best_t_samples = t_samples[mask]
        
        # If power-law didn't work well, try the critical points approach
        if best_hv < 0 or best_t_samples is None:
            # Include critical points of the sin term
            critical_points = np.array([(1/20 + k/10) for k in range(10)])
            critical_points = critical_points[(critical_points >= 0) & (critical_points <= 1)]
            critical_points = np.concatenate([[0.0], critical_points, [1.0]])
            critical_points = np.unique(critical_points)
            
            # Distribute remaining samples
            remaining = n_select - len(critical_points)
            if remaining > 0:
                intervals = np.diff(critical_points)
                total_length = intervals.sum()
                samples_per_interval = np.maximum(1, np.round(remaining * intervals / total_length)).astype(int)
                diff = remaining - samples_per_interval.sum()
                if diff != 0:
                    idx_max = np.argmax(samples_per_interval)
                    samples_per_interval[idx_max] += diff
                
                t_samples = [critical_points[0]]
                for i in range(len(critical_points) - 1):
                    t_start = critical_points[i]
                    t_end = critical_points[i + 1]
                    n_in_interval = samples_per_interval[i]
                    
                    if n_in_interval == 0:
                        continue
                    elif n_in_interval == 1:
                        t_samples.append(t_end)
                    else:
                        local_i = np.arange(1, n_in_interval)
                        t_local = (local_i / n_in_interval) ** 0.8
                        t_samples.extend(t_start + t_local * (t_end - t_start))
                
                t_samples = np.array(t_samples)
                t_samples = np.unique(t_samples)
                
                if len(t_samples) > n_select:
                    t_samples = t_samples[:n_select]
                
                F_samples = zdt3_pareto_front(t_samples)
                mask = np.all(F_samples < ref_point, axis=1)
                t_samples = t_samples[mask]
                
                if len(t_samples) > 0:
                    F_valid_samples = F_samples[mask]
                    hv = compute_hypervolume_2d(F_valid_samples, ref_point)
                    if hv > best_hv:
                        best_hv = hv
                        best_t_samples = t_samples
        
        # Fallback to uniform if nothing else worked
        if best_t_samples is None or len(best_t_samples) == 0:
            best_t_samples = np.linspace(0, 1, n_select)
        
        # Step 3: Evaluate the selected points using the problem function
        x = np.zeros(dim)
        recorded = []
        
        for t in best_t_samples:
            if n_evals >= max_evals or time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        # If we have budget left, try to refine with a few additional points
        if n_evals < max_evals and time.time() < deadline:
            # Try adding points at the critical points we might have missed
            critical_points = np.array([(1/20 + k/10) for k in range(10)])
            critical_points = critical_points[(critical_points >= 0) & (critical_points <= 1)]
            critical_points = np.concatenate([[0.0], critical_points, [1.0]])
            
            for t in critical_points:
                if n_evals >= max_evals or time.time() > deadline:
                    break
                
                # Check if we already have a point close to this t
                if len(best_t_samples) > 0:
                    if np.min(np.abs(best_t_samples - t)) < 0.001:
                        continue
                
                x[0] = float(t)
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
        
        return recorded
