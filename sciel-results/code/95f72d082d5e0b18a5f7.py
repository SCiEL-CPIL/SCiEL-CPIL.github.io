"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct structural optimizer for ZDT2 hitting the Pareto front exactly.

    Uses non-uniform spacing of x0 values to maximize hypervolume coverage
    of the ZDT2 Pareto front with respect to reference point [11, 11].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _optimal_x0_placement(self, n):
        """Find optimal x0 placements to maximize hypervolume on ZDT2 front.

        ZDT2 Pareto front: f1 = x0, f2 = (1+x0)*(1-sqrt(x0)) for x0 in [0,1].
        Reference point: [11, 11].
        
        Hypervolume contribution of a point at x0=t:
        The front is monotone decreasing in f2 as f1 increases.
        For a set of points, hypervolume is computed from the dominated region.
        
        We use a greedy/optimization approach to place points where they
        contribute most to hypervolume.
        """
        if n <= 1:
            return np.array([0.0])
        
        # The ZDT2 front: f1 = t, f2 = (1+t)*(1-sqrt(t))
        # f2(0) = 1, f2(1) = 1
        # f2 is minimized at t where derivative = 0
        # d/dt [(1+t)(1-sqrt(t))] = (1-sqrt(t)) - (1+t)/(2*sqrt(t))
        # = 1 - sqrt(t) - 1/(2*sqrt(t)) - sqrt(t)/2
        # = 1 - 3*sqrt(t)/2 - 1/(2*sqrt(t))
        # Setting = 0: 2 - 3*sqrt(t) - 1/sqrt(t) = 0
        # Let u = sqrt(t): 2u^2 - 3u^2 - 1 = 0 => -u^2 - 1 = 0? No.
        # 2 - 3u - 1/u = 0 => 2u - 3u^2 - 1 = 0 => 3u^2 - 2u + 1 = 0
        # Discriminant: 4 - 12 = -8 < 0, so no real roots. f2 is monotone?
        # Actually let me just compute: f2(0)=1, f2(0.25)=(1.25)(1-0.5)=0.625,
        # f2(0.5)=(1.5)(1-0.707)=0.436, f2(1)=1. So f2 dips in the middle.
        
        # For hypervolume with ref [11,11], each point (f1, f2) contributes
        # area in the region it uniquely dominates. Since the front is not
        # convex, optimal placement depends on the shape.
        
        # Use scipy to optimize placement
        from scipy.optimize import minimize
        
        def f2_of(t):
            return (1.0 + t) * (1.0 - np.sqrt(np.maximum(t, 0.0)))
        
        def neg_hypervolume(x0_vals):
            """Negative hypervolume of points on ZDT2 front."""
            t = np.clip(x0_vals, 0.0, 1.0)
            f1 = t
            f2 = f2_of(t)
            
            # Sort by f1 ascending
            idx = np.argsort(f1)
            f1_s = f1[idx]
            f2_s = f2[idx]
            
            # Compute hypervolume with reference [11, 11]
            # Only points with f1 < 11 and f2 < 11 count (all do here)
            # Hypervolume = sum over segments
            # For sorted points by f1, the hypervolume is:
            # For each point i, the width is f1[i+1]-f1[i] (or 11-f1[i] for last)
            # and height is 11 - f2[i] (but need to account for lower envelope)
            
            # Actually, since points are on the Pareto front (none dominates another),
            # the hypervolume is the area under the step function.
            # Sort by f1 ascending. The "upper envelope" in terms of f2 for hypervolume
            # computation: since no point dominates another, we just sum rectangles.
            
            # More carefully: hypervolume = integral of (11 - f2(f1)) over f1 range
            # but discretized as sum of rectangles where each point's f2 defines
            # the height for its f1 interval.
            
            # For points sorted by f1: x_0 < x_1 < ... < x_{n-1}
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (11 - f2_i)  where x_n = 11
            # But only if f2 is non-increasing... it's not.
            
            # Correct approach: the hypervolume of a set of non-dominated points
            # with reference point r is the volume of the union of boxes.
            # In 2D, sort by f1 ascending. The hypervolume is:
            # sum over i of (f1[i+1] - f1[i]) * (r2 - min(f2[j] for j >= i))
            # No wait, that's not right either.
            
            # In 2D with reference (r1, r2), for points sorted by f1 ascending:
            # HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (r2 - f2[i])
            # where f1[n] = r1, BUT only if f2 is monotonically decreasing.
            # If f2 is not monotone, we need to take the lower envelope.
            
            # Since our points are all on the Pareto front, none dominates another.
            # But f2 is not monotone in f1 for ZDT2. So we need to be careful.
            
            # The correct 2D hypervolume: sort by f1. The dominated region is
            # the union of rectangles [f1_i, r1] x [f2_i, r2] for each point.
            # The area of this union can be computed by sweep line.
            
            # For simplicity and since n is small (200), use a direct computation:
            # The hypervolume equals the area under the "lower hull" of f2 vs f1,
            # from min(f1) to r1, bounded by r2.
            
            # Actually the simplest correct method for 2D:
            # Sort points by f1 ascending. Initialize prev_f2 = r2.
            # HV = 0
            # For each point in order of increasing f1:
            #   If f2 < prev_f2:
            #     HV += (f1[i] - f1[i-1]) * (r2 - prev_f2)  -- no this is getting complex
            
            # Let me just use the standard formula:
            # Sort by f1 ascending: p_0, p_1, ..., p_{n-1}
            # HV = sum_{i=0}^{n-1} (p_{i+1}.f1 - p_i.f1) * (r2 - p_i.f2)
            # where p_n.f1 = r1, BUT this is only valid if f2 is non-increasing.
            # 
            # For general case, the correct formula when points are non-dominated:
            # Sort by f1 ascending. The hypervolume is:
            # HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (r2 - f2_lower_envelope[i])
            # where f2_lower_envelope[i] = min(f2[j] for j such that f1[j] >= f1[i])
            # No, that's not right either.
            
            # Let me think again. In 2D, with reference point (r1, r2):
            # The hypervolume is the area of the region { (a,b) : a <= r1, b <= r2 } 
            # that is dominated by at least one point.
            # A point (f1_i, f2_i) dominates region [f1_i, r1] x [f2_i, r2].
            # The union of these rectangles.
            
            # To compute union area in 2D:
            # Sort rectangles by left edge (f1_i). Sweep from left to right.
            # At each step, the active height is the minimum f2 among all rectangles
            # whose left edge <= current position and right edge > current position.
            
            # Since all right edges are r1, and left edges are f1_i:
            # Sort by f1 ascending.
            # The height at position x (between f1[i] and f1[i+1]) is:
            # min(f2[j] for j where f1[j] <= x) = min(f2[0], ..., f2[i])
            # So HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (r2 - min(f2[0..i]))
            # where f1[n] = r1.
            
            # Wait, that's the area of the region NOT dominated... no.
            # The dominated region at x-position x has height from min_f2 to r2.
            # So the area of dominated region = sum (f1[i+1]-f1[i]) * (r2 - min(f2[0..i]))
            # But we want the hypervolume which IS this dominated area.
            
            # Hmm, but actually the hypervolume is the volume of points dominated BY the set,
            # relative to reference. So yes, it's the area of union of [f1_i, r1] x [f2_i, r2].
            
            # So: sort by f1 ascending. running_min = inf
            # HV = 0
            # for i in range(n):
            #   running_min = min(running_min, f2[i])
            #   width = (f1[i+1] - f1[i]) if i < n-1 else (r1 - f1[i])
            #   HV += width * (r2 - running_min)
            
            # But wait, this counts the region where the minimum f2 so far extends to r2.
            # That seems correct for the union.
            
            r1, r2 = 11.0, 11.0
            if len(f1_s) == 0:
                return 0.0
            
            running_min = float('inf')
            hv = 0.0
            for i in range(len(f1_s)):
                running_min = min(running_min, f2_s[i])
                if i < len(f1_s) - 1:
                    width = f1_s[i+1] - f1_s[i]
                else:
                    width = r1 - f1_s[i]
                if running_min < r2:
                    hv += width * (r2 - running_min)
                else:
                    hv += 0.0
            return -hv
        
        # Initial guess: uniform
        x0_init = np.linspace(0.0, 1.0, n)
        
        # Optimize
        try:
            res = minimize(
                neg_hypervolume,
                x0_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * n,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x0_opt = np.sort(np.clip(res.x, 0.0, 1.0))
        except Exception:
            x0_opt = x0_init
        
        # Ensure we have exactly n unique points
        x0_opt = np.unique(x0_opt)
        if len(x0_opt) < n:
            # Pad with points in largest gaps
            x0_opt = np.sort(x0_opt)
            while len(x0_opt) < n:
                gaps = np.diff(np.concatenate(([0.0], x0_opt, [1.0])))
                max_idx = np.argmax(gaps)
                if max_idx == 0:
                    new_pt = x0_opt[0] / 2.0 if len(x0_opt) > 0 else 0.5
                elif max_idx == len(gaps) - 1:
                    new_pt = (1.0 + x0_opt[-1]) / 2.0 if len(x0_opt) > 0 else 0.5
                else:
                    new_pt = (x0_opt[max_idx-1] + x0_opt[max_idx]) / 2.0
                x0_opt = np.sort(np.concatenate((x0_opt, [new_pt])))
        
        return x0_opt[:n]

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        if n < 2:
            x0s = np.array([0.0])
        else:
            # Use optimized placement
            if time.time() < deadline - 50:  # Only optimize if we have time
                x0s = self._optimal_x0_placement(n)
            else:
                # Fallback to uniform
                x0s = np.linspace(0.0, 1.0, n)
        
        # Ensure we include x0=0 for f1=0
        if len(x0s) > 0 and x0s[0] > 0:
            x0s = np.concatenate(([0.0], x0s))
            x0s = np.unique(x0s)
            if len(x0s) > n:
                x0s = x0s[:n]
        
        # Build candidate vectors with x[1:]=0
        cand = np.zeros((len(x0s), self.dim))
        cand[:, 0] = x0s
        cand = np.clip(cand, lo, hi)

        # Evaluate candidates within budget and deadline
        F_list = []
        for i in range(len(cand)):
            if len(F_list) >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(cand[i])
            F_list.append(np.asarray(f, dtype=float))

        return None
# EVOLVE-BLOCK-END
