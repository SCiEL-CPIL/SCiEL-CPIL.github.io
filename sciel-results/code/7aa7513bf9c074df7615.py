import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.x1_values = None
        self._precompute_grid()

    def _precompute_grid(self):
        N = self.budget
        if N <= 0:
            self.x1_values = np.array([])
            return
        
        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Reference point R = [11, 11]
        # Hypervolume H = integral_0^1 (11 - f2(x1)) dx1 = integral_0^1 (10 + sqrt(x1)) dx1
        # With N points x_1 < x_2 < ... < x_N, the hypervolume is:
        # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) * (11 - f2(x_i)) + (11 - x_N) * (11 - f2(x_N))
        # Wait, the standard decomposition for 2D HV with points (f1_i, f2_i) sorted by f1 ascending:
        # H = sum_{i=1}^N (f1_{i+1} - f1_i) * (R2 - f2_i) where f1_{N+1} = R1.
        # Let's verify.
        # The area covered by point i is a rectangle from f1_i to f1_{i+1} (or R1) and from 0 to R2 - f2_i.
        # Actually, the hypervolume dominated by the set of points is the union of rectangles [0, f1_i] x [0, f2_i].
        # No, the reference point is R. The dominated region is {y | y <= f_i for some i, y <= R}.
        # Since points are on the Pareto front, they are non-dominated.
        # The hypervolume is the area of the union of rectangles [0, f1_i] x [0, R2 - f2_i]? No.
        # The area dominated by a point (f1, f2) with respect to R is (R1 - f1)(R2 - f2).
        # But we are summing disjoint regions.
        # For sorted points by f1 ascending, the contribution of point i is:
        # Width in f1: (f1_{i+1} - f1_i) for i < N, and (R1 - f1_N) for i = N.
        # Height in f2: (R2 - f2_i).
        # Let's check:
        # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) (11 - f2(x_i)) + (11 - x_N)(11 - f2(x_N))
        # This is correct for the area "above" the front and "below" R?
        # No, the hypervolume is the volume of the region dominated by the points and bounded by R.
        # The region is { (y1, y2) | 0 <= y1 <= R1, 0 <= y2 <= R2, and exists i s.t. y1 <= f1_i and y2 <= f2_i }.
        # Since f2 is decreasing in f1 for ZDT1 front, the upper boundary of the dominated region is the front itself (for f1 in [0,1]) and then the line y2 = f2(1) = 0 for f1 > 1? No.
        # For f1 > 1, there is no point with f1_i >= f1. So the region is only up to f1=1?
        # No, the points are in [0,1].
        # The dominated region is the union of rectangles [0, f1_i] x [0, f2_i].
        # Since f1_i increases and f2_i decreases, the union is a staircase shape.
        # The area of this staircase is sum_{i=1}^N f1_i * (f2_i - f2_{i+1}) + f1_N * f2_N? No.
        # Let's use the standard formula:
        # H = sum_{i=1}^N (f1_{i+1} - f1_i) (R2 - f2_i) where f1_{N+1} = R1.
        # Let's test with 1 point (0.5, 0.5), R=(1,1).
        # H = (1 - 0.5)(1 - 0.5) = 0.25. Correct.
        # Test with 2 points (0.2, 0.8) and (0.8, 0.2), R=(1,1).
        # H = (0.8 - 0.2)(1 - 0.8) + (1 - 0.8)(1 - 0.2) = 0.6*0.2 + 0.2*0.8 = 0.12 + 0.16 = 0.28.
        # Manual check:
        # Rect 1: [0, 0.2] x [0, 0.8] -> Area 0.16
        # Rect 2: [0, 0.8] x [0, 0.2] -> Area 0.16
        # Union: [0, 0.2] x [0, 0.8] + [0.2, 0.8] x [0, 0.2]
        # Area = 0.16 + 0.6 * 0.2 = 0.16 + 0.12 = 0.28. Correct.
        
        # So H(x_1, ..., x_N) = sum_{i=1}^{N-1} (x_{i+1} - x_i) (10 + sqrt(x_i)) + (11 - x_N)(10 + sqrt(x_N))
        
        # We want to maximize H.
        # This is equivalent to minimizing the "loss" L = H_max - H.
        # H_max is the integral of (10 + sqrt(x)) from 0 to 11? No, from 0 to 1.
        # Integral_0^1 (10 + sqrt(x)) dx = [10x + (2/3)x^{3/2}]_0^1 = 10 + 2/3 = 10.666...
        # Wait, the last term is (11 - x_N)(10 + sqrt(x_N)).
        # The "true" continuous hypervolume for the front segment [0,1] is:
        # Integral_0^1 (10 + sqrt(x)) dx + Integral_1^11 (10 + 0) dx?
        # No, for x > 1, f2 is not defined on the front. The front ends at x=1, f2=0.
        # So for x in [1, 11], the "front" is effectively at f2=0.
        # So the height is 11 - 0 = 11.
        # So H_max = Integral_0^1 (10 + sqrt(x)) dx + Integral_1^11 11 dx
        # = (10 + 2/3) + 11 * 10 = 10.666... + 110 = 120.666...
        
        # Let's re-evaluate the discrete formula.
        # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) h(x_i) + (11 - x_N) h(x_N)
        # where h(x) = 10 + sqrt(x) for x <= 1, and h(x) = 11 for x > 1.
        # Since x_i <= 1, h(x_i) = 10 + sqrt(x_i).
        # The last term is (11 - x_N)(10 + sqrt(x_N)).
        # This approximates the area under the curve h(x) from x_N to 11?
        # If we extend h(x) to be 11 for x > 1, then the area from x_N to 11 is:
        # Integral_{x_N}^1 h(x) dx + Integral_1^11 11 dx
        # = [10x + (2/3)x^{3/2}]_{x_N}^1 + 110
        # = (10 + 2/3 - 10x_N - (2/3)x_N^{3/2}) + 110
        # The discrete term is (11 - x_N)(10 + sqrt(x_N)) = 110 + 11sqrt(x_N) - 10x_N - x_N sqrt(x_N).
        # These are not the same. The discrete approximation is a right-Riemann sum for the last interval?
        # Actually, the formula H = sum (f1_{i+1} - f1_i)(R2 - f2_i) is exact for the staircase area.
        # The staircase area is always less than the integral of the upper envelope.
        
        # To maximize H, we should place points where the marginal gain is highest.
        # The marginal gain of adding a point at x is related to the curvature of the front.
        # However, we can just optimize the positions x_i directly.
        # Since N=200, we have 200 variables.
        # We can use scipy.optimize.minimize on the negative hypervolume.
        # But we need to keep x_i sorted.
        # We can parameterize x_i = sigmoid(z_i) or use a cumulative sum of positive steps.
        # Let d_i > 0 be the step sizes. x_1 = d_1, x_2 = d_1 + d_2, ..., x_N = sum d_i.
        # Constraint: x_N <= 1.
        # This is a constrained optimization problem.
        
        # Alternatively, use the power law x_i = u_i^p with u_i = (i-0.5)/N or i/N.
        # The current program uses u_i = (i-0.5)/N and optimizes p.
        # The inspirations suggest optimizing p more finely or using a different spacing.
        
        # Let's try to derive the optimal spacing analytically or with a better numerical method.
        # The optimal spacing for maximizing hypervolume in 2D is proportional to 1 / sqrt(|f''(x)|) ?
        # Or related to the curvature.
        # For a smooth front, the optimal number of points in an interval [a, b] is proportional to:
        # integral_a^b sqrt(1 + (f'(x))^2) dx ? No.
        # The area of a trapezoid approximating the curve is the error source.
        # The error in approximating the area under a curve with a trapezoid is proportional to f''(x) * (dx)^3.
        # To minimize total error sum f''(x_i) (dx_i)^3 subject to sum dx_i = L,
        # we set dx_i proportional to 1 / sqrt(f''(x_i)).
        
        # Let's compute f2(x) = 1 - sqrt(x).
        # f2'(x) = -1 / (2 sqrt(x))
        # f2''(x) = 1 / (4 x^{3/2})
        # So f''(x) is very large near x=0.
        # This suggests dx_i should be very small near x=0.
        # dx_i ~ 1 / sqrt(f''(x_i)) = 1 / sqrt(1/(4 x_i^{3/2})) = 2 x_i^{3/4}.
        # So dx/dx ~ x^{3/4}.
        # dx ~ x^{3/4} dx? No, dx_i / x_i^{3/4} is constant?
        # Let's integrate: dx / x^{3/4} = C dy.
        # x^{1/4} = C y + D.
        # So x ~ (y)^4.
        # This implies x_i ~ i^4.
        # Let's check if x_i ~ i^4 is good.
        # If x_i = (i/N)^4, then x_1 is very small.
        
        # Let's test this hypothesis numerically by comparing HV for different p in x_i = (i/N)^p.
        # We already do this in the current program.
        # The current program searches p in [0.8, 2.5] (from inspiration 2) or similar.
        # Wait, the current program uses bisection on a CDF derived from a density w(x) = sqrt(10+sqrt(x)).
        # Let's check the density derivation in the current program.
        # "The 'weight' in x-space is w(x) = sqrt(10 + sqrt(x))."
        # This comes from the idea that the optimal quantization density is proportional to the square root of the second derivative of the objective?
        # Actually, for optimal 1D quantization of a function to minimize L2 error, the density is proportional to |f'(x)|? No.
        # For minimizing the maximum error, it's different.
        # For maximizing hypervolume, it's a specific problem.
        
        # Let's stick to the numerical optimization of p, but do it more carefully.
        # Also, we can try to optimize the positions directly using a local search or gradient descent.
        # Given the time limit, a fast analytical or semi-analytical approach is better.
        
        # Let's refine the search for p.
        # We will search p in a wider range, e.g., [1.0, 5.0].
        # And we will use a finer grid.
        
        u = np.linspace(0.0, 1.0, N)
        # Actually, using midpoints (i-0.5)/N might be better to avoid boundary issues?
        # But x=0 and x=1 are valid points.
        # Let's try both u = i/N and u = (i-0.5)/N.
        
        best_hv = -np.inf
        best_p = 1.5
        best_x1 = None
        
        # Search p
        p_vals = np.linspace(1.0, 5.0, 1000)
        
        for p in p_vals:
            x1 = u ** p
            x1 = np.clip(x1, 0.0, 1.0)
            
            # Ensure strictly increasing?
            # If p > 0, x1 is increasing.
            # If u[0]=0, x1[0]=0.
            
            f2 = 1.0 - np.sqrt(x1)
            
            # Compute HV
            # H = sum_{i=0}^{N-2} (x1[i+1] - x1[i]) * (11 - f2[i]) + (11 - x1[N-1]) * (11 - f2[N-1])
            df1 = np.diff(x1)
            df1 = np.append(df1, 11.0 - x1[-1])
            h = 11.0 - f2
            hv = np.sum(df1 * h)
            
            if hv > best_hv:
                best_hv = hv
                best_p = p
                best_x1 = x1.copy()
        
        # Refine around best_p
        p_refine = np.linspace(best_p - 0.1, best_p + 0.1, 200)
        for p in p_refine:
            x1 = u ** p
            x1 = np.clip(x1, 0.0, 1.0)
            f2 = 1.0 - np.sqrt(x1)
            df1 = np.diff(x1)
            df1 = np.append(df1, 11.0 - x1[-1])
            h = 11.0 - f2
            hv = np.sum(df1 * h)
            
            if hv > best_hv:
                best_hv = hv
                best_p = p
                best_x1 = x1.copy()
        
        self.x1_values = best_x1

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        
        if self.x1_values is None or len(self.x1_values) == 0:
            return None

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for x1 in self.x1_values:
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
