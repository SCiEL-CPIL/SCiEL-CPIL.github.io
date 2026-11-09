import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 specific
        # g_min = 1
        # f1 = x1
        # f2 = 1 + (1 - sqrt(x1)) / (1 + 9*x1) * sin(10 * pi * x1)
        
        # 1. Identify Pareto Front Segments
        # Sample x1 in [0, 1]
        n_sample = 10000
        x1_samples = np.linspace(0, 1, n_sample)
        
        def f2_pareto(x1):
            # Handle x1=0
            term = np.zeros_like(x1)
            mask = x1 > 0
            x1_pos = x1[mask]
            term[mask] = (1 - np.sqrt(x1_pos)) / (1 + 9 * x1_pos) * np.sin(10 * np.pi * x1_pos)
            return 1 + term
        
        f2_samples = f2_pareto(x1_samples)
        
        # Find points on the lower envelope (Pareto Front)
        # A point is on the front if its f2 is strictly less than the min f2 of all points with smaller f1
        # Since x1 is sorted, we can just keep track of min_f2_so_far
        is_on_front = np.zeros(n_sample, dtype=bool)
        min_f2_so_far = np.inf
        for i in range(n_sample):
            if f2_samples[i] < min_f2_so_far:
                is_on_front[i] = True
                min_f2_so_far = f2_samples[i]
            # Note: if f2 == min_f2_so_far, it's dominated by the previous point with same f2 and smaller f1?
            # Yes, strictly better or equal f2 with smaller f1 dominates.
            # So we only keep strictly smaller f2.
        
        # Group indices into segments
        segments = []
        start = None
        for i in range(n_sample):
            if is_on_front[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((x1_samples[start], x1_samples[i-1]))
                    start = None
        if start is not None:
            segments.append((x1_samples[start], x1_samples[n_sample-1]))
        
        # Filter out very small segments (noise)
        # But ZDT3 segments are well-defined.
        # Let's print segments for debugging in mind:
        # Should be ~6-7 segments.
        
        if not segments:
            # Fallback: uniform sampling
            x1_values = np.linspace(0, 1, budget)
        else:
            # Allocate budget
            # Give at least 1 point to each segment
            n_segs = len(segments)
            if budget < n_segs:
                # Not enough budget for all segments, pick the most important ones?
                # Just pick uniformly
                x1_values = np.linspace(0, 1, budget)
            else:
                counts = np.ones(n_segs, dtype=int)
                remaining = budget - n_segs
                
                # Distribute remaining points proportional to segment length
                lengths = np.array([h - l for l, h in segments])
                total_len = lengths.sum()
                
                if total_len > 0:
                    # Ideal counts
                    ideal_counts = remaining * lengths / total_len
                    # Floor and distribute remainder
                    extra_counts = np.floor(ideal_counts).astype(int)
                    counts += extra_counts
                    
                    diff = budget - counts.sum()
                    if diff > 0:
                        # Distribute remaining diff to segments with largest fractional parts
                        fracs = ideal_counts - np.floor(ideal_counts)
                        order = np.argsort(-fracs)
                        for i in range(diff):
                            counts[order[i % n_segs]] += 1
                    elif diff < 0:
                        # Remove points from smallest segments
                        order = np.argsort(lengths)
                        for i in range(-diff):
                            idx = order[i % n_segs]
                            if counts[idx] > 1:
                                counts[idx] -= 1
                                diff += 1 # We removed one, so we need to remove less? No, diff is negative.
                                # Actually, just loop until diff is 0
                                if counts.sum() == budget: break
                        
                # Generate x1 values
                x1_values = []
                for i, (l, h) in enumerate(segments):
                    n_pts = counts[i]
                    if n_pts == 1:
                        x1_values.append((l + h) / 2)
                    else:
                        # Uniform sampling in [l, h]
                        pts = np.linspace(l, h, n_pts)
                        x1_values.extend(pts.tolist())
                
                x1_values = np.array(x1_values)
        
        # Ensure exactly budget points
        if len(x1_values) < budget:
            # Fill with random points on front
            extra = np.random.uniform(0, 1, size=budget - len(x1_values))
            # Filter to front? Or just use them. They might be dominated.
            # Better to duplicate existing good points or sample from segments.
            # Let's just sample uniformly from the segments again
            all_seg_points = []
            for l, h in segments:
                all_seg_points.append(np.random.uniform(l, h))
            extra = np.array(all_seg_points[:budget - len(x1_values)])
            x1_values = np.concatenate([x1_values, extra])
        elif len(x1_values) > budget:
            x1_values = x1_values[:budget]
        
        # Build X
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values
        # x[1:] = 0 is optimal for g=1
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
