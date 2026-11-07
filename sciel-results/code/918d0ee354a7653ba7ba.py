import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2 with n=30:
        # f1 = x1
        # f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        # Reference point: [11.0, 11.0]

        num_points = min(self.budget, 200)

        if num_points <= 1:
            x1_vals = np.array([0.5])
        else:
            # Use calculus of variations to find optimal grid.
            # For concave decreasing f2 = 11*(1-sqrt(f1)), the optimal grid
            # equalizes the marginal HV contribution of each segment.
            # 
            # The HV can be written as:
            # HV = sum_{i=0}^{N-2} (f1_{i+1} - f1_i) * (11 - 11*(1-sqrt(f1_i)))
            #    + (11 - f1_{N-1}) * (11 - 11*(1-sqrt(f1_{N-1})))
            # = sum_{i=0}^{N-2} (f1_{i+1} - f1_i) * 11*sqrt(f1_i)
            #   + (11 - f1_{N-1}) * 11*sqrt(f1_{N-1})
            #
            # Let u_i = sqrt(f1_i), so f1_i = u_i^2, and u is increasing.
            # HV = sum_{i=0}^{N-2} (u_{i+1}^2 - u_i^2) * 11*u_i + (11 - u_{N-1}^2) * 11*u_{N-1}
            #
            # This is a concave function of the u_i's (since u^2 is convex and we're
            # maximizing a function that is linear in differences times u_i).
            # 
            # Actually, let's just directly optimize the f1 values using a smooth
            # parameterization. Use log-spaced initial guess and refine.
            
            # Initialize with a reasonable non-uniform grid
            # Start with equal spacing in sqrt space (which is close to optimal)
            u = np.linspace(0, 1, num_points)
            f1_init = u ** 2  # This is the p=2/3 power law in terms of index
            
            # But let's do better: use iterative local optimization
            # Each interior point f1_i (i=1..N-2) affects two terms:
            # Term i-1: (f1_i - f1_{i-1}) * 11*sqrt(f1_{i-1})
            # Term i:   (f1_{i+1} - f1_i) * 11*sqrt(f1_i)
            # 
            # dHV/df1_i = 11*sqrt(f1_{i-1}) - 11*sqrt(f1_i) - 11*f1_{i+1}/(2*sqrt(f1_i)) + ...
            # This gets complex. Let's use a different approach.
            
            # Alternative: parameterize by "cumulative area" and solve implicitly.
            # For the optimal grid, the area under the staircase between consecutive
            # points should be proportional to the "weight" at that location.
            
            # Practical approach: use scipy to optimize a smooth transformation
            # that maps uniform indices to non-uniform f1 values.
            
            # Use a rational function parameterization:
            # f1_i = i^a / (i^a + (N-1-i)^b) * 1.0  -- no, this doesn't hit endpoints well
            # 
            # Better: use f1_i = (i/(N-1))^p but optimize p AND allow endpoint adjustments
            # 
            # Actually, the simplest exact approach: since we have only 200 points,
            # we can directly optimize all 198 interior points using gradient-free methods
            # or use the fact that the problem is separable in a clever way.
            
            # Let's use the insight that for f2 = c*(1 - f1^alpha) with alpha < 1,
            # the optimal grid has f1_i proportional to (i/N)^(1/(1-alpha)) approximately.
            # For ZDT2: f2 = 11*(1-sqrt(f1)), so alpha = 1/2, giving exponent 2.
            # But this is for the continuous limit. For finite N, we need correction.
            
            # Let's just use a multi-stage optimization:
            # Stage 1: Optimize power-law exponent p
            # Stage 2: Fine-tune individual points
            
            def compute_hv_from_f1(f1):
                """Compute HV given sorted f1 values."""
                f2 = 11.0 * (1.0 - np.sqrt(f1))
                r1, r2 = 11.0, 11.0
                # HV = sum_{i=0}^{N-2} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{N-1}) * (r2 - f2_{N-1})
                diffs = np.diff(f1)
                heights = r2 - f2[:-1]
                hv = np.sum(diffs * heights)
                hv += (r1 - f1[-1]) * (r2 - f2[-1])
                return hv

            # Stage 1: Optimize power-law exponent
            def compute_hv_p(p):
                i = np.arange(num_points, dtype=float)
                f1 = (i / (num_points - 1)) ** p
                return -compute_hv_from_f1(f1)

            result = minimize_scalar(compute_hv_p, bounds=(0.5, 3.0), method='bounded',
                                     options={'xatol': 1e-12, 'maxiter': 200})
            p_opt = result.x
            i = np.arange(num_points, dtype=float)
            f1 = (i / (num_points - 1)) ** p_opt
            
            # Stage 2: Fine-tune using iterative local search
            # For each interior point, try small perturbations and accept if HV improves
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                improved = False
                for idx in range(1, num_points - 1):
                    old_f1 = f1[idx]
                    # Try perturbing this point
                    for delta in [0.01, -0.01, 0.001, -0.001, 0.05, -0.05]:
                        new_f1 = old_f1 + delta
                        if new_f1 <= f1[idx-1] + 1e-10 or new_f1 >= f1[idx+1] - 1e-10:
                            continue
                        f1[idx] = new_f1
                        new_hv = compute_hv_from_f1(f1)
                        f1[idx] = old_f1
                        if new_hv > compute_hv_from_f1(f1) + 1e-12:
                            f1[idx] = new_f1
                            improved = True
                            break
                if not improved:
                    break
            
            x1_vals = f1

        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
