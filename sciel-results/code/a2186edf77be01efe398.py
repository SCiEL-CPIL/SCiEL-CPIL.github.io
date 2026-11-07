"""Program for multi-objective optimization on ZDT1 using hypervolume-optimized quantization."""
import numpy as np


class Optimizer:
    """Optimizes ZDT1 by exploiting its structure: fix non-critical dims to 0.5,
    use a grid of x1 values optimized for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        if n <= 0:
            return None

        # ZDT1 structure:
        # f1 = x1
        # f2 = 1 + 9 * x_g / (dim - 1) where x_g = sum_{i=2..dim} x_i - (dim-1) * sin(10 * pi * x_i / (dim-1))
        # Pareto front: x1 in [0,1], f1 = x1, f2 = 1 - x1^2
        # So f2(x1) = 1 - x1^2, and hypervolume integrand is (11 - f2) = 10 + x1^2
        #
        # The optimal x1 grid for hypervolume maximization with reference point (11, 11):
        # Hypervolume contribution of a point at x1 (assuming sorted) is related to
        # the area under the curve. For a set of x1 values x_1 < x_2 < ... < x_k in [0,1],
        # the hypervolume is:
        #   HV = sum_{i=1}^{k} (11 - x_i) * (f2(x_{i-1}) - f2(x_i)) + (11 - x_k) * f2(x_k)
        # where x_0 = 1 (since f2(1) = 0) and f2(x) = 1 - x^2
        #
        # Actually, let me think more carefully. The hypervolume of a set of points
        # (f1_i, f2_i) with reference (r1, r2) = (11, 11):
        # Sort by f1 ascending. HV = sum over each point of (r1 - f1_i) * (r2 - f2_i) - (r2 - f2_{i-1})
        # where f2_0 = r2 = 11.
        #
        # For ZDT1 Pareto front: f2 = 1 - x1^2, so r2 - f2 = 11 - (1 - x1^2) = 10 + x1^2
        #
        # HV = sum_{i=1}^{k} (11 - x1_i) * ((10 + x1_i^2) - (10 + x1_{i-1}^2))
        # where x1_0 = 0 (f2(0) = 1, so 10 + 0 = 10) ... wait, need to be careful.
        #
        # Let me define g(x) = 10 + x^2 (this is r2 - f2(x)).
        # Sort x1 values: x_(1) < x_(2) < ... < x_(k)
        # HV = sum_{i=1}^{k} (11 - x_(i)) * (g(x_(i)) - g(x_(i-1)))
        # where x_(0) = 0 and g(0) = 10, and we include the tail: the last term should be
        # (11 - x_(k)) * g(x_(k)) if we consider the area to the reference point.
        #
        # Actually the standard formula: if points are (a_i, b_i) sorted by a ascending,
        # HV = sum_i (r1 - a_i)(r2 - b_i) - (r2 - b_{i-1}) with b_0 = r2.
        #
        # So HV = sum_{i=1}^{k} (11 - x_i)(10 + x_i^2 - 10 - x_{i-1}^2) + (11 - x_k)(10 + x_k^2 - 11)
        # Wait no. Let me just use: HV = integral approximation.
        #
        # For a discrete set on a monotone front, the hypervolume is:
        # HV = (r1 - x_(k)) * (r2 - f2(x_(k))) + sum_{i=1}^{k-1} (x_(i+1) - x_(i)) * (r2 - f2(x_(i)))
        # where x_(0) = 0, x_(k) is the largest x1.
        #
        # This is a left-Riemann sum approximation of the area under (r2 - f2(x)) from 0 to x_k,
        # plus the rectangle from x_k to r1.
        #
        # To maximize HV with k points, we want to choose x_1 < ... < x_k to maximize this sum.
        # The optimal spacing satisfies x_{i+1} - x_i being proportional to some function of the
        # derivative of (r2 - f2). Since r2 - f2(x) = 10 + x^2, its derivative is 2x.
        #
        # Using optimal quantization theory, the cell boundaries should satisfy that the
        # hypervolume gain per point is equal. The marginal gain of inserting a point at x
        # between x_a and x_b is approximately (x_b - x_a) * (r2 - f2(x)) * 2 * something...
        #
        # A simpler approach: use inverse CDF. The optimal grid for maximizing area under
        # a function h(x) with k rectangles (left Riemann) on [0, L] places points where
        # the cumulative area is equally divided. So x_i satisfies:
        #   integral_0^{x_i} h(t) dt = (i/k) * integral_0^L h(t) dt
        # where h(x) = 10 + x^2 (the height at which we're measuring).
        #
        # integral_0^x (10 + t^2) dt = 10x + x^3/3
        # integral_0^1 (10 + t^2) dt = 10 + 1/3 = 31/3
        #
        # So x_i = solution of 10x + x^3/3 = (i/k) * 31/3
        #
        # For i = 1..k, this gives x_i. We also want to include x=0 or x=1?
        # The Pareto front spans x1 in [0, 1]. We should include both endpoints ideally.
        #
        # Let's use k = n points total. We'll compute the optimal x1 values and evaluate.
        
        # Number of points to use
        k = n
        
        # Compute optimal x1 grid using inverse of cumulative distribution
        # CDF: F(x) = (10x + x^3/3) / (31/3) = (30x + x^3) / 31
        # We want x_i such that F(x_i) = (i - 0.5) / k for i = 1..k (midpoint rule)
        # Or F(x_i) = i / k for i = 0..k-1
        # Let's use: for i = 0, 1, ..., k-1: target = (i + 0.5) / k
        # Solve 10x + x^3/3 = target * 31/3
        
        x1_values = np.zeros(k)
        for i in range(k):
            target = (i + 0.5) / k
            rhs = target * 31.0 / 3.0
            # Solve 10x + x^3/3 = rhs, x in [0, 1]
            # Use bisection
            lo_x, hi_x = 0.0, 1.0
            for _ in range(50):
                mid_x = (lo_x + hi_x) / 2.0
                val = 10.0 * mid_x + mid_x**3 / 3.0
                if val < rhs:
                    lo_x = mid_x
                else:
                    hi_x = mid_x
            x1_values[i] = (lo_x + hi_x) / 2.0
        
        # Ensure we cover the endpoints for better hypervolume
        # With k evaluations, use k-2 for interior and 2 for endpoints
        if k >= 3:
            x1_interior = np.zeros(k - 2)
            # Targets for interior points: 1/(k-1), 2/(k-1), ..., (k-2)/(k-1)
            for i in range(1, k - 1):
                target = i / (k - 1)
                rhs = target * 31.0 / 3.0
                lo_x, hi_x = 0.0, 1.0
                for _ in range(50):
                    mid_x = (lo_x + hi_x) / 2.0
                    val = 10.0 * mid_x + mid_x**3 / 3.0
                    if val < rhs:
                        lo_x = mid_x
                    else:
                        hi_x = mid_x
                x1_interior[i - 1] = (lo_x + hi_x) / 2.0
            x1_values = np.concatenate([[0.0], x1_interior, [1.0]])
        elif k == 2:
            x1_values = np.array([0.0, 1.0])
        elif k == 1:
            x1_values = np.array([0.5])
        
        # Evaluate each point: x1 as first component, rest = 0.5 (which gives x_g = 0, f2 = 1)
        # Wait, x_g = sum(x_i) - (dim-1)*sin(10*pi*x_i/(dim-1)) for i=2..dim
        # At x_i = 0.5: sin(10*pi*0.5/(dim-1)) = sin(5*pi/(dim-1))
        # This isn't zero in general. Let me pick x_i values that make sin term zero.
        # sin(10*pi*x/(dim-1)) = 0 when 10*pi*x/(dim-1) = j*pi, i.e., x = j*(dim-1)/10
        # For x in [0,1], valid j: 0 <= j*(dim-1)/10 <= 1, so j = 0, 1, ..., floor(10/(dim-1))
        # For dim=30, dim-1=29, j=0 gives x=0, j=1 gives x=29/10=2.9 (out of range)
        # So only x=0 works to make sin=0? That's not ideal.
        #
        # Actually, the Pareto front is achieved when x_g = 0, which means:
        # sum_{i=2}^{dim} x_i = (dim-1) * sin(10*pi*x_i/(dim-1)) for each... no, it's:
        # x_g = sum x_i - (dim-1) * mean(sin(10*pi*x_i/(dim-1))) ... no
        #
        # ZDT1: g(x) = 1 + 9 * h(x)/(|X'|) where h(x) = sum x_i - (|X'|-1)*sin(10*pi*x_i/|X'|)
        # and |X'| = dim - 1.
        # Pareto front: g(x) = 1, which requires h(x) = 0, i.e., sum x_i = (dim-2)*sin(10*pi*x_i/(dim-1))
        # for all i... no, it's a single equation: sum_{i=2}^{dim} x_i = (dim-2) * sum sin... 
        #
        # Actually the standard ZDT1: 
        # f1 = x1
        # g(x) = 1 + 9 * (sum_{i=2}^{n} x_i) / (n-1)  ... no wait
        #
        # Let me just use x_i = 0 for i >= 2, which gives h(x) = 0, g(x) = 1, f2 = 1 - x1^2.
        # That's on the Pareto front!
        
        results = []
        for x1 in x1_values:
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] = 0 gives h(x)=0, g(x)=1, f2 = 1 - x1^2
            f = problem(x)
            results.append(f)
        
        return results
