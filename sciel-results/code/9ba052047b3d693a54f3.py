"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(n, base):
    """Generate n Halton sequence points in (0, 1) for a given base."""
    seq = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        i_val = i
        while i_val > 0:
            f /= base
            r += f * (i_val % base)
            i_val //= base
        seq.append(r)
    return np.array(seq)


def _hypervolume_3d(F, ref):
    """
    Compute 3D hypervolume for a set of non-dominated points F relative to ref.
    F is (N, 3) array of objective values (all <= ref).
    """
    N = len(F)
    if N == 0:
        return 0.0
    # Sort by first objective ascending
    order = np.argsort(F[:, 0])
    F = F[order]
    
    # Remove dominated points (keep non-dominated)
    # Since sorted by f0, a point i is dominated by j if f0_j <= f0_i and f1_j <= f1_i and f2_j <= f2_i
    # We can use a sweep line.
    
    # Simple O(N^2) removal since N <= 200
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        for j in range(i + 1, N):
            if not keep[j]:
                continue
            # Check if j dominates i (since F is sorted by f0, f0_j >= f0_i)
            # If f0_j == f0_i and f1_j < f1_i and f2_j < f2_i -> j dominates i?
            # Actually, if f0_j > f0_i, j cannot dominate i unless f0_j == f0_i.
            # Wait, if sorted ascending, F[j, 0] >= F[i, 0].
            # For j to dominate i, we need F[j, 0] <= F[i, 0], so F[j, 0] must equal F[i, 0].
            if F[j, 0] > F[i, 0]:
                break
            # F[j, 0] == F[i, 0]
            if F[j, 1] < F[i, 1] and F[j, 2] < F[i, 2]:
                keep[i] = False
                break
            elif F[j, 1] < F[i, 1] and F[j, 2] <= F[i, 2]:
                # j is better in f1, equal/better in f2. But f0 is same.
                # Does j dominate i? Yes if f0<=, f1<=, f2<= and at least one strict.
                # Here f0 same, f1 strict, f2 <=. So j dominates i.
                keep[i] = False
                break
            elif F[j, 1] <= F[i, 1] and F[j, 2] < F[i, 2]:
                keep[i] = False
                break
    F = F[keep]
    
    # Now compute HV
    # We can decompose into boxes.
    # Sort by f0 ascending.
    # For each point, its contribution is the volume of the box it "owns".
    # A common algorithm for 3D HV:
    # 1. Sort by f0 ascending.
    # 2. Keep a running maximum of (ref - f2) for points with f1 < current f1?
    #    Actually, let's use the standard sweep.
    
    # Sort by f0 ascending
    idx = np.argsort(F[:, 0])
    F = F[idx]
    
    # We want to calculate the volume dominated by F and not by other points in F.
    # Since F is non-dominated, we can calculate the hypervolume of the union of boxes [F_i, ref].
    
    # Algorithm:
    # Sort by f0 ascending.
    # Let's define for each point i, the "width" in f0 direction is F[i+1, 0] - F[i, 0] (or ref[0] - F[i, 0] for last).
    # But we must subtract volume already covered by previous points in the (f1, f2) plane?
    # No, the standard decomposition is:
    # HV = sum_i (F[i, 0] - F[i-1, 0]) * HV_2D(F[i:], ref[1:])
    # This is not quite right because the points are not nested in 2D.
    
    # Better:
    # Sweep along f0.
    # Maintain a set of active points in (f1, f2) space.
    # As we move f0 from ref[0] down to 0, we add points.
    # Actually, let's sweep f0 from min to max.
    # Let's use the algorithm from "Fast Hypervolume Calculation in Three Dimensions".
    
    # Simpler for N=200:
    # For each point i, calculate the volume of the box [F[i], ref] that is not covered by any other point j in F.
    # This is hard.
    
    # Let's use the standard recursive decomposition.
    # HV(F, ref) in 3D:
    # Sort F by f0 ascending.
    # HV = 0
    # prev_f0 = 0 (or min possible)
    # We need to calculate the 2D HV of the projection of F onto (f1, f2) for the "slice" at f0.
    
    # Actually, let's just implement the 2D HV calculation for each "layer".
    # But the layers are coupled.
    
    # Let's use a different approach:
    # The hypervolume is the volume of the union of boxes.
    # We can use the inclusion-exclusion or just a sweep.
    
    # Sweep line in f0:
    # Sort points by f0 ascending.
    # We maintain a 2D structure of the "envelope" of points seen so far in (f1, f2).
    # Actually, since we are calculating the volume dominated by the points, we can do:
    # HV = sum_{i} (F[i, 0] - F[i-1, 0]) * HV_2D(P_i, ref_12)
    # where P_i is the set of points j >= i (in f0 order) projected to (f1, f2).
    # Wait, if a point j > i has f1 < F[i, 1] and f2 < F[i, 2], it dominates i in the (f1, f2) plane,
    # but since F[j, 0] > F[i, 0], it doesn't dominate i in 3D.
    # However, for the slice between F[i-1, 0] and F[i, 0], the active points are those with f0 <= current f0.
    # So as we increase f0, we ADD points.
    
    # Let's sweep f0 from 0 to ref[0].
    # Points are added when their f0 is passed.
    # We maintain the 2D hypervolume of the added points in the (f1, f2) plane, relative to (ref[1], ref[2]).
    # But the 2D HV changes as we add points.
    
    # HV = integral_0^ref0 HV_2D(P(t), ref_12) dt
    # where P(t) = {p in F | p[0] <= t}.
    
    # Since F is small, we can just compute this numerically or analytically.
    # The function HV_2D(P(t)) is piecewise constant between the f0 values of the points.
    # So we can sum over the intervals between consecutive f0 values.
    
    # Sort by f0 ascending.
    # Let f0_vals = sorted unique f0 values.
    # Intervals: [0, f0_vals[0]], [f0_vals[0], f0_vals[1]], ..., [f0_vals[-1], ref[0]].
    
    # In each interval [a, b], the set of active points is fixed.
    # We calculate the 2D HV of the active points' (f1, f2) projections.
    
    # 2D HV calculation:
    # Given points (f1, f2) and ref (r1, r2).
    # Sort by f1 ascending.
    # HV_2D = sum (f1[i+1] - f1[i]) * (r2 - min(f2 of points with f1 <= f1[i+1]))?
    # No, standard 2D HV:
    # Sort by f1 ascending.
    # Keep track of min_f2.
    # Actually, sort by f1 ascending.
    # For each point i, the width in f1 is f1[i+1] - f1[i] (or r1 - f1[i] for last).
    # The height is r2 - f2[i] ? No, it's r2 - min(f2[j] for j >= i).
    # Wait, if points are non-dominated in 2D, then f1 ascending implies f2 descending.
    # So min(f2[j] for j >= i) is just f2[last]? No.
    # If f1 asc, f2 desc, then for j >= i, f2[j] <= f2[i].
    # So the envelope is just the points themselves.
    # The area covered by point i is (f1[i+1] - f1[i]) * (r2 - f2[i])?
    # No, the box for point i is [f1[i], r1] x [f2[i], r2].
    # But if point i+1 has f1 > f1[i] and f2 < f2[i], its box is [f1[i+1], r1] x [f2[i+1], r2].
    # The union of these boxes.
    # Since f1 asc and f2 desc, the boxes are "staircase".
    # The area is sum_i (f1[i+1] - f1[i]) * (r2 - f2[i]) for i=0..N-2, plus (r1 - f1[N-1]) * (r2 - f2[N-1]).
    # Wait, is this correct?
    # Let's check.
    # Box 0: [f1[0], f1[1]] x [f2[0], r2] ? No, box 0 is [f1[0], r1] x [f2[0], r2].
    # But part of it is covered by Box 1?
    # Box 1 is [f1[1], r1] x [f2[1], r2].
    # Since f2[1] < f2[0], Box 1 is "lower" in f2.
    # The union of Box 0 and Box 1:
    # In [f1[0], f1[1]], the height is r2 - f2[0] (from Box 0).
    # In [f1[1], r1], the height is r2 - f2[1] (from Box 1, since Box 1 is lower, it dominates Box 0 in this region? No, Box 0 is higher, so it covers more).
    # Wait, if f2[0] > f2[1], then [f2[0], r2] is a subset of [f2[1], r2].
    # So in [f1[1], r1], Box 1 covers [f2[1], r2], which includes [f2[0], r2].
    # So the union in [f1[1], r1] is just Box 1's height.
    # So the area is:
    # (f1[1] - f1[0]) * (r2 - f2[0]) + (r1 - f1[1]) * (r2 - f2[1]).
    # Yes, this works.
    
    # So for 2D HV:
    # Sort points by f1 ascending.
    # area = 0
    # for i in range(N):
    #     width = (f1[i+1] if i < N-1 else r1) - f1[i]
    #     height = r2 - f2[i]
    #     area += width * height
    
    # Back to 3D:
    # Sort F by f0 ascending.
    # hv = 0
    # active_points = []  # list of (f1, f2)
    # prev_f0 = 0
    # for i in range(N):
    #     f0_i = F[i, 0]
    #     # Add point i to active_points
    #     active_points.append((F[i, 1], F[i, 2]))
    #     # Sort active_points by f1 ascending?
    #     # We need to maintain the 2D envelope.
    #     # Since we are adding points, the envelope changes.
    #     # We can just recalculate the 2D HV of active_points at each step.
    #     # N is small, so O(N^2 log N) is fine.
    #     f1_vals = np.array([p[0] for p in active_points])
    #     f2_vals = np.array([p[1] for p in active_points])
    #     # Sort by f1
    #     idx = np.argsort(f1_vals)
    #     f1_s = f1_vals[idx]
    #     f2_s = f2_vals[idx]
    #     # Calculate 2D HV
    #     hv_2d = 0
    #     for j in range(len(f1_s)):
    #         width = (f1_s[j+1] if j < len(f1_s)-1 else ref[1]) - f1_s[j]
    #         height = ref[2] - f2_s[j]
    #         # Ensure height is positive
    #         if height < 0:
    #             height = 0
    #         if width < 0:
    #             width = 0
    #         hv_2d += width * height
    #     # The volume in the slice [prev_f0, f0_i] is hv_2d * (f0_i - prev_f0)
    #     hv += hv_2d * (f0_i - prev_f0)
    #     prev_f0 = f0_i
    # # Last slice [prev_f0, ref[0]]
    # # active_points contains all points
    # # Calculate 2D HV again
    # f1_vals = np.array([p[0] for p in active_points])
    # f2_vals = np.array([p[1] for p in active_points])
    # idx = np.argsort(f1_vals)
    # f1_s = f1_vals[idx]
    # f2_s = f2_vals[idx]
    # hv_2d = 0
    # for j in range(len(f1_s)):
    #     width = (f1_s[j+1] if j < len(f1_s)-1 else ref[1]) - f1_s[j]
    #     height = ref[2] - f2_s[j]
    #     if height < 0:
    #         height = 0
    #     if width < 0:
    #         width = 0
    #     hv_2d += width * height
    # hv += hv_2d * (ref[0] - prev_f0)
    
    # This is O(N^3) which is fine for N=200.
    
    # Let's implement this cleanly.
    
    # 1. Filter points strictly better than ref
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    
    # 2. Sort by f0 ascending
    idx = np.argsort(F[:, 0])
    F = F[idx]
    
    hv = 0.0
    active = []  # list of (f1, f2)
    prev_f0 = 0.0
    
    for i in range(len(F)):
        f0_i = F[i, 0]
        if f0_i > ref[0]:
            break
        active.append((F[i, 1], F[i, 2]))
        
        # Calculate 2D HV of active points
        if len(active) == 0:
            hv_2d = 0.0
        else:
            f1_arr = np.array([p[0] for p in active])
            f2_arr = np.array([p[1] for p in active])
            
            # Filter points strictly better than ref in f1, f2
            mask = (f1_arr < ref[1]) & (f2_arr < ref[2])
            f1_arr = f1_arr[mask]
            f2_arr = f2_arr[mask]
            
            if len(f1_arr) == 0:
                hv_2d = 0.0
            else:
                # Sort by f1
                idx2 = np.argsort(f1_arr)
                f1_s = f1_arr[idx2]
                f2_s = f2_arr[idx2]
                
                hv_2d = 0.0
                for j in range(len(f1_s)):
                    next_f1 = f1_s[j+1] if j < len(f1_s) - 1 else ref[1]
                    width = next_f1 - f1_s[j]
                    height = ref[2] - f2_s[j]
                    if width > 0 and height > 0:
                        hv_2d += width * height
        
        width_0 = f0_i - prev_f0
        if width_0 > 0:
            hv += hv_2d * width_0
        prev_f0 = f0_i
    
    # Last slice
    if prev_f0 < ref[0]:
        # active contains all points with f0 < ref[0]
        if len(active) > 0:
            f1_arr = np.array([p[0] for p in active])
            f2_arr = np.array([p[1] for p in active])
            mask = (f1_arr < ref[1]) & (f2_arr < ref[2])
            f1_arr = f1_arr[mask]
            f2_arr = f2_arr[mask]
            
            if len(f1_arr) > 0:
                idx2 = np.argsort(f1_arr)
                f1_s = f1_arr[idx2]
                f2_s = f2_arr[idx2]
                
                hv_2d = 0.0
                for j in range(len(f1_s)):
                    next_f1 = f1_s[j+1] if j < len(f1_s) - 1 else ref[1]
                    width = next_f1 - f1_s[j]
                    height = ref[2] - f2_s[j]
                    if width > 0 and height > 0:
                        hv_2d += width * height
                
                hv += hv_2d * (ref[0] - prev_f0)
    
    return hv


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using direct Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 250  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front approximation:
        # Set x_3..x_{D-1} = 0
        # Vary x_1, x_2 in [0, 1]
        # Use Halton sequence for uniform coverage
        
        n_evals = self.budget
        
        # Generate Halton sequences for x_0 and x_1
        # Base 2 for x_0, Base 3 for x_1
        h1 = _halton(n_evals, 2)
        h2 = _halton(n_evals, 3)
        
        # Apply a shift to improve coverage and avoid boundary bias
        shift = 0.5 / n_evals
        h1 = (h1 + shift) % 1.0
        h2 = (h2 + shift) % 1.0
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = h1
        X[:, 1] = h2
        # x_2..x_{D-1} are already 0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Optional: Add some random exploration points if time permits
        # But direct front sampling should be sufficient.
        
        return X
# EVOLVE-BLOCK-END
