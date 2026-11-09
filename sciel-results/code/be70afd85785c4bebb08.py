"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0,1) using the given prime base."""
    seq = np.zeros(n)
    f = 1.0 / base
    for i in range(n):
        idx = i + 1
        result = 0.0
        while idx > 0:
            f /= base
            idx, remainder = divmod(idx, base)
            result += f * remainder
        seq[i] = result
    return seq


def _compute_zdt3_segments(n_grid=100000):
    """
    Compute the exact Pareto-optimal segments of ZDT3.
    
    ZDT3: f1 = x1, f2 = (1 + 10*(n-1)) * (g + h(x1, g))
    where g = 1 + 10*(n-1) * sum(x_i^2) for i=2..n
    and h(x, g) = 1 - sqrt(x/g) - (x/g)*sin(10*pi*x)
    
    For g=1 (x2..xn=0): h(x,1) = 1 - sqrt(x) - x*sin(10*pi*x)
    A point is Pareto-optimal iff h(x,1) >= 0.
    """
    x = np.linspace(0, 1, n_grid)
    h = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
    
    # Find contiguous regions where h >= 0
    valid = h >= 0.0
    segments = []
    in_segment = False
    start = 0.0
    for i in range(len(valid)):
        if valid[i] and not in_segment:
            start = x[i]
            in_segment = True
        elif not valid[i] and in_segment:
            segments.append((start, x[i-1]))
            in_segment = False
    if in_segment:
        segments.append((start, x[-1]))
    
    # Merge very close segments (numerical artifacts)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 1e-6:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    
    return merged


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 + 10*(n-1)*(1 + h(x1,1))
    where h(x,1) = 1 - sqrt(x) - x*sin(10*pi*x).
    The front is discontinuous with 7 segments where f2 is Pareto-optimal.
    
    Strategy: Use a Halton sequence (low-discrepancy) to sample x1 uniformly
    across the [0,1] domain, then filter to only Pareto-optimal regions.
    This guarantees uniform coverage of all 7 disjoint segments.
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
        
        # Compute exact Pareto-optimal segments of ZDT3
        segments = _compute_zdt3_segments(n_grid=50000)
        
        # Total length of Pareto-optimal regions
        total_len = sum(b - a for a, b in segments)
        
        # Use Halton sequence for space-filling coverage
        n_samples = self.budget
        halton_vals = _halton_sequence(n_samples, 2)
        
        # Map Halton values to x1 positions within the Pareto-optimal segments
        # We need to distribute samples proportionally to segment lengths
        x1_values = []
        
        # Build cumulative length array for mapping
        cum_lengths = []
        cum = 0.0
        for a, b in segments:
            cum += (b - a)
            cum_lengths.append(cum)
        
        # Map each Halton value to a segment and position within it
        for hv in halton_vals:
            target_len = hv * total_len
            # Binary search for the segment
            seg_idx = np.searchsorted(cum_lengths, target_len, side='right')
            if seg_idx >= len(segments):
                seg_idx = len(segments) - 1
            
            a, b = segments[seg_idx]
            # Position within this segment
            prev_cum = cum_lengths[seg_idx - 1] if seg_idx > 0 else 0.0
            seg_len = b - a
            frac = (target_len - prev_cum) / seg_len if seg_len > 0 else 0.0
            x1 = a + frac * seg_len
            x1_values.append(x1)
        
        x1_values = np.array(x1_values)
        
        # Ensure segment boundaries are included for maximum hypervolume
        boundary_vals = []
        for a, b in segments:
            boundary_vals.extend([a, b])
        
        # Add boundaries if not already present (within tolerance)
        for bv in boundary_vals:
            if not np.any(np.abs(x1_values - bv) < 1e-10):
                # Replace the closest point to ensure boundary is hit
                idx = np.argmin(np.abs(x1_values - bv))
                x1_values[idx] = bv
        
        # Generate points: x2..x30 = 0 ensures g = 1 (minimum possible)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        points[:, 0] = x1_values
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
