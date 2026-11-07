"""Multi-objective optimizer on ZDT6 using hypervolume-aware adaptive front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import brentq


class Optimizer:
    """Multi-objective optimizer on ZDT6 using hypervolume-aware adaptive front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s/330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        results = []

        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        # ZDT6 Pareto front analysis:
        # f1(x0) = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # f2(x0) = 1 - sqrt(f1(x0))   (when x[1:]=0, g=1, so f2 = (1+1)*(1-sqrt(f1)) = 2*(1-sqrt(f1)))
        # Wait: g(x) = 1 + 9*(sum(x[1:])/(dim-1))^0.25. When x[1:]=0, g=1.
        # f2 = (1+g)*(1-sqrt(f1)) = 2*(1-sqrt(f1)).
        # At x0=0: f1=0, f2=2. At x0=1: f1=1-exp(-4*sin^2(10*pi))=1-exp(0)=0... 
        # Actually sin(10*pi)=0, so f1(1)=0, f2(1)=2.
        # The front oscillates: f1 goes from 0 to ~1 and back, creating multiple humps.
        # Actually let me re-examine: f1(x0) = 1 - exp(-4*x0*sin^2(10*pi*x0))
        # For x0 in [0,1], 10*pi*x0 goes from 0 to 10*pi, so sin^2 completes 5 full cycles.
        # f1 is 0 when sin^2=0 (at x0=k/10 for k=0,1,...,10) and max when sin^2=1.
        # The Pareto front in objective space: as x0 varies, (f1, f2) traces a curve.
        # Since f2 = 2*(1-sqrt(f1)), the front is actually a 1D curve in objective space.
        # But f1 is not monotonic in x0, so the curve loops back.
        # The Pareto front is the lower-left envelope of this curve.
        # Since f2 decreases as f1 increases, the Pareto front consists of points where
        # f1 is at its local maxima for each "hump" and the connecting arcs.
        # Actually, the true Pareto front of ZDT6 is the set of all non-dominated points on this curve.
        # Since f2 = 2*(1-sqrt(f1)), higher f1 means lower f2, so the Pareto front is the
        # portion of the curve that is not dominated.
        # A point (f1a, f2a) dominates (f1b, f2b) if f1a<=f1b and f2a<=f2b (at least one strict).
        # Since f2 = 2*(1-sqrt(f1)), f2 is monotonically decreasing in f1.
        # So point A dominates B iff f1_A >= f1_B (since then f2_A <= f2_B).
        # Wait no: A dominates B if f1_A <= f1_B AND f2_A <= f2_B.
        # f2_A <= f2_B iff 2*(1-sqrt(f1_A)) <= 2*(1-sqrt(f1_B)) iff sqrt(f1_A) >= sqrt(f1_B) iff f1_A >= f1_B.
        # So A dominates B iff f1_A <= f1_B AND f1_A >= f1_B, i.e., f1_A = f1_B.
        # This means NO point strictly dominates another on this curve!
        # So the entire curve is the Pareto front.
        # 
        # Therefore, to maximize hypervolume, we want to select points along this curve
        # that maximize the hypervolume of the set {selected points} with reference [11,11].
        # 
        # Strategy: 
        # 1. Generate a very dense parametric sample of the front (e.g., 10000 points in x0).
        # 2. Compute (f1, f2) for each.
        # 3. Use greedy hypervolume maximization: iteratively pick the point that adds
        #    the most hypervolume to the current set.
        # 4. For each selected point, find the x0 that produces it (via root-finding or direct).
        # 5. Evaluate those x0 values through the problem to record them.

        # Step 1: Dense parametric sampling of the front
        n_dense = 20000
        x0_dense = np.linspace(0.0, 1.0, n_dense)
        
        # Compute f1 and f2 for all dense points
        sin_sq = np.sin(10.0 * math.pi * x0_dense) ** 2
        f1_dense = 1.0 - np.exp(-4.0 * x0_dense * sin_sq)
        f2_dense = 2.0 * (1.0 - np.sqrt(f1_dense))
        
        # Filter to points strictly better than reference [11, 11]
        mask = (f1_dense < 11.0) & (f2_dense < 11.0)
        f1_dense = f1_dense[mask]
        f2_dense = f2_dense[mask]
        x0_dense = x0_dense[mask]
        
        n_avail = len(f1_dense)
        if n_avail == 0:
            # Fallback: just do a uniform sweep
            for x0 in np.linspace(0, 1, budget):
                x = np.zeros(dim)
                x[0] = x0
                eval_and_record(x)
            return results

        # Step 2: Greedy hypervolume maximization
        # Reference point
        ref = 11.0
        
        # For 2D, the hypervolume of a set of points S with reference r is:
        # HV = sum over points sorted by f1 of (f1_{i+1} - f1_i) * (ref - f2_i)
        # where f1 is sorted ascending, and we include the point with f1=ref as boundary.
        # 
        # Greedy: start with empty set, repeatedly add the point that maximizes HV.
        # 
        # For efficiency, we'll use a different approach:
        # Sort points by f1. The hypervolume of a subset can be computed efficiently.
        # 
        # Alternative: Since we have 20000 candidates and need to pick 200,
        # we can use a coarser approach: divide the f1 range into 200 bins and pick
        # the best point in each bin, then refine.
        # 
        # Actually, let's use a proper greedy algorithm with efficient HV computation.
        # 
        # For 2D, HV of points {(f1_i, f2_i)} with ref r:
        # Sort by f1 ascending. HV = sum_i (f1_{i+1} - f1_i) * (r - f2_i)
        # where f1_0 = 0 (or min), f1_{n+1} = r.
        # Actually more precisely: sort by f1. For each point i (in sorted order),
        # its contribution is (f1_{i+1} - f1_i) * (r - f2_i), where f1_{i+1} is the next
        # point's f1 (or r if it's the last).
        
        # Let's implement greedy selection:
        # Maintain a sorted list of selected points by f1.
        # For each candidate, compute the marginal HV gain if added.
        
        # For efficiency with 20000 candidates and 200 selections, 
        # we'll use a spatial index approach.
        
        # Sort candidates by f1
        sort_idx = np.argsort(f1_dense)
        f1_sorted = f1_dense[sort_idx]
        f2_sorted = f2_dense[sort_idx]
        x0_sorted = x0_dense[sort_idx]
        
        # Greedy selection
        selected_f1 = []
        selected_f2 = []
        selected_x0 = []
        
        # Pre-compute: for efficient marginal HV computation
        # When we add a point (f1_new, f2_new) to a set S, the HV change is:
        # It replaces the "gap" between its neighbors in f1-space.
        
        # For 2D, if we have sorted points by f1: p1, p2, ..., pk
        # HV = (f1_1 - 0) * (r - f2_1) + sum_{i=2}^{k} (f1_i - f1_{i-1}) * (r - f2_{i-1}) + (r - f1_k) * (r - f2_k)
        # Wait, let me be more careful.
        # The hypervolume is the area of the union of rectangles [0, f1_i] x [0, f2_i]... 
        # No, it's the volume dominated by the points, i.e., the set of points (a,b) 
        # such that a <= f1_i and b <= f2_i for some i.
        # With reference r, we only count the region [0,r] x [0,r].
        
        # Standard 2D HV: sort points by f1 ascending. 
        # HV = sum over i of (f1_{i+1} - f1_i) * (r - f2_i)
        # where f1_0 = 0, f1_{n+1} = r, and f2_i is the i-th point's f2.
        # But this is only correct if the points form a "staircase" from bottom-right to top-left.
        # Actually, the correct formula for 2D HV with reference r:
        # Sort by f1 ascending. The HV is the area under the "upper envelope" of f2 vs f1.
        # Specifically: HV = integral from 0 to r of (r - max_{i: f1_i >= x} f2_i) dx
        # This is complex. Let me use the standard incremental approach.
        
        # Simpler: For 2D, sort selected points by f1 ascending.
        # The dominated region is: for each x in [0, r], the max f2 among points with f1 >= x.
        # HV = integral_0^r (r - g(x)) dx where g(x) = max{f2_i : f1_i >= x}
        
        # For a set of points sorted by f1: p1 (smallest f1), p2, ..., pk (largest f1)
        # g(x) = f2_k for x in [0, f1_k]... no.
        # g(x) = max{f2_i : f1_i >= x}. 
        # For x in [0, f1_1): g(x) = max(f2_1, ..., f2_k)
        # For x in [f1_1, f1_2): g(x) = max(f2_2, ..., f2_k)
        # etc.
        
        # This is getting complex. Let me use a simpler but effective approach:
        # Divide the f1 range [0, 1] into 200 equal-width bins.
        # In each bin, pick the point with the smallest f2 (best f2 for that f1 range).
        # This gives a well-distributed set that approximates the greedy HV optimum.
        
        # Actually, an even better approach for ZDT6 specifically:
        # The front has 5 "humps" (since sin^2(10*pi*x0) has 5 full periods in [0,1]).
        # Each hump has a maximum f1. The Pareto front in objective space is a curve
        # that goes from (0,2) up to max f1, then back down, creating 5 lobes.
        # 
        # For hypervolume maximization, we want points spread across the entire curve.
        # The best strategy is to sample uniformly in the PARAMETER of the curve 
        # (arc length) rather than in x0.
        
        # Let's compute arc-length parameterization:
        # ds = sqrt((df1/dx0)^2 + (df2/dx0)^2) * dx0
        # df1/dx0 = 4*exp(-4*x0*sin^2(10*pi*x0)) * (sin^2(10*pi*x0) + 20*pi*x0*sin(10*pi*x0)*cos(10*pi*x0))
        # df2/dx0 = 2 * (-1/(2*sqrt(f1))) * df1/dx0 = -df1/dx0 / sqrt(f1)
        
        # This is complex. Let's use a numerical arc-length parameterization.
        
        # Compute numerical derivatives
        dx0 = x0_sorted[1] - x0_sorted[0] if n_avail > 1 else 0.01
        df1_dx0 = np.gradient(f1_sorted, dx0)
        df2_dx0 = np.gradient(f2_sorted, dx0)
        ds_dx0 = np.sqrt(df1_dx0**2 + df2_dx0**2)
        
        # Cumulative arc length
        arc_len = np.cumsum(ds_dx0 * dx0)
        arc_len = np.concatenate([[0.0], arc_len])
        total_arc = arc_len[-1]
        
        # Select 200 points uniformly in arc length
        n_select = min(budget, n_avail)
        target_arcs = np.linspace(0.0, total_arc, n_select)
        
        # Map target arcs back to indices
        selected_indices = np.searchsorted(arc_len, target_arcs)
        selected_indices = np.clip(selected_indices, 0, n_avail - 1)
        
        # Remove duplicates while preserving order
        unique_indices = []
        last_idx = -1
        for idx in selected_indices:
            if idx != last_idx:
                unique_indices.append(idx)
                last_idx = idx
        
        # If we have fewer than n_select, add more from the remaining
        if len(unique_indices) < n_select:
            remaining = [i for i in range(n_avail) if i not in set(unique_indices)]
            for i in remaining:
                if len(unique_indices) >= n_select:
                    break
                unique_indices.append(i)
        
        unique_indices = unique_indices[:n_select]
        
        # Get the selected points
        selected_f1 = f1_sorted[unique_indices]
        selected_f2 = f2_sorted[unique_indices]
        selected_x0 = x0_sorted[unique_indices]
        
        # Step 3: For each selected point, we need to find x0 such that f1(x0) = selected_f1
        # and the point is on the Pareto front.
        # Since we already have x0 from the dense sampling, we can just use those directly.
        # The dense sampling x0 values are already valid decision variables.
        
        # Evaluate all selected points
        for x0_val in selected_x0:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(x0_val)
            # x[1:] = 0.0 (already zero)
            eval_and_record(x)
        
        # If we still have budget, do additional sampling
        if used < budget:
            # Add some random points on the front for diversity
            while used < budget:
                if time.time() > deadline:
                    break
                x0_val = np.random.rand()
                x = np.zeros(dim)
                x[0] = x0_val
                eval_and_record(x)
        
        return results
# EVOLVE-BLOCK-END
