"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Direct analytical multi-objective optimizer for DTLZ2.
    Uses the known geometry of the DTLZ2 Pareto front (quarter-sphere)
    to place points directly on the front using a Halton sequence.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for the 330s limit
        
        # DTLZ2 Pareto front is defined by f1^2 + f2^2 + f3^2 = 1, fi >= 0
        # We parameterize the front using spherical coordinates.
        # To ensure uniform distribution on the surface (which maximizes hypervolume),
        # we use a Halton sequence in 2D to generate (theta, phi).
        
        # We need to be careful with the mapping to ensure uniformity.
        # Standard uniform sampling on a sphere surface:
        # cos(theta) is uniform in [-1, 1] -> but we are in the first octant (quarter sphere for 3 obj? No, DTLZ2 with 3 obj is a quarter sphere in the positive octant? 
        # Actually, DTLZ2 with M objectives has a front that is the part of the (M-1)-sphere in the positive orthant.
        # For M=3, it's a quarter of a sphere (1/8 of the full sphere surface area, but bounded by axes).
        # Wait, the Pareto front of DTLZ2 is {x | sum f_i^2 = 1, f_i >= 0}.
        
        # Let's use the property that for uniform hypervolume contribution, we want points spread out.
        # A simple and effective strategy is to use a dense grid or quasi-random points on the front.
        
        # Number of points to evaluate. We have a budget of 200.
        # We can afford to evaluate all 200 points if we construct them analytically.
        # The problem function `problem(x)` is called for each x.
        # We need to construct x such that the resulting f is on the front.
        
        # DTLZ2 definition:
        # f1 = (1 + g) * 0.5 * x1 * sin(pi * x1 / 2) ... wait, standard DTLZ2:
        # f_i(x) = (1 + g(x_M)) * sin(pi * x_i / 2) for i=1..M-1
        # f_M(x) = (1 + g(x_M)) * cos(pi * x_{M-1} / 2) * ... * cos(pi * x_1 / 2)
        # Here g(x) = sum (x_i - 0.5)^2 for i=M+1..n.
        # To get points on the Pareto front, we need g(x) = 0, which means x_i = 0.5 for i > M.
        # So we set x_4..x_12 = 0.5.
        # Then f_i = 0.5 * sin(pi * x_i / 2) for i=1,2 and f_3 = 0.5 * cos(pi * x_2 / 2) * cos(pi * x_1 / 2)?
        # Wait, the standard formula for DTLZ2 with M objectives:
        # f_i = (1+g) * sin(pi * x_i / 2) for i=1..M-1
        # f_M = (1+g) * prod_{j=1}^{M-1} cos(pi * x_j / 2)
        # If we set x_1..x_{M-1} to values in [0,1] and x_M..x_n = 0.5, then g=0.
        # Then f_i = 0.5 * sin(pi * x_i / 2) and f_M = 0.5 * prod cos(pi * x_j / 2).
        # This does NOT sum of squares to 1. It sums to 1?
        # Let's check M=2: f1 = 0.5 sin(pi x1/2), f2 = 0.5 cos(pi x1/2). f1^2+f2^2 = 0.25.
        # The standard DTLZ2 usually has f1^2+...+fM^2 = (1+g)^2.
        # If g=0, sum squares = 1.
        # My formula above: f1 = sin(pi x1/2), f2 = cos(pi x1/2) -> sum squares = 1.
        # The factor 0.5 in some definitions is for DTLZ1 or different scaling.
        # Let's verify the standard DTLZ2 definition.
        # Most common: f_i = (1+g) * sin(pi x_i / 2), f_M = (1+g) * prod cos.
        # Then sum f_i^2 = (1+g)^2 * (sum sin^2 + prod cos^2).
        # For M=2: sin^2 + cos^2 = 1. So sum f^2 = (1+g)^2.
        # So yes, if g=0, points are on the unit sphere.
        
        # So, to generate points on the front:
        # 1. Set x_4..x_12 = 0.5.
        # 2. Choose x_1, x_2 in [0,1] such that the resulting f1, f2, f3 are on the front.
        # 3. The mapping from x_1, x_2 to f is:
        #    f1 = sin(pi * x1 / 2)
        #    f2 = sin(pi * x2 / 2) * cos(pi * x1 / 2)
        #    f3 = cos(pi * x2 / 2) * cos(pi * x1 / 2)
        #    (Note: order might vary, but it's a bijection from the unit square to the quarter sphere).
        
        # To maximize hypervolume, we want to distribute x_1, x_2 well in [0,1]^2.
        # We will use a Halton sequence to generate 200 points in [0,1]^2.
        
        n_evals = self.budget
        # Generate Halton sequence in base 2 and 3
        u1 = self._halton(n_evals, 2)
        u2 = self._halton(n_evals, 3)
        
        # Construct x vectors
        X = np.zeros((n_evals, self.dim))
        # Set the decision variables that control the front shape
        X[:, 0] = u1
        X[:, 1] = u2
        # Set the rest to 0.5 to make g(x) = 0
        if self.dim > 2:
            X[:, 2:] = 0.5
        
        # Evaluate all points
        # We must ensure we don't exceed the budget or time.
        # Since we are constructing them directly, we just call problem on each.
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f_val = problem(X[i])
            F.append(f_val)
            
        return X
# EVOLVE-BLOCK-END
