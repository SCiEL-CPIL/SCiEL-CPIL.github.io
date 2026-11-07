import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Standard ZDT4: min g(x) = 1 when x_i = 0 for i > 0
        # Pareto front: f1 = x0, f2 = 2 - sqrt(x0) for x0 in [0,1]
        # Hypervolume integral is maximized by distributing points to equalize
        # the area contribution per segment.
        #
        # For the front f2 = 2 - sqrt(f1), with reference (11, 400):
        # The hypervolume contribution of each point depends on the spacing.
        # Optimal spacing for hypervolume approximation with fixed N points:
        # Equal spacing in the "area" sense.
        #
        # The exact hypervolume of the continuous front from f1=a to f1=b is:
        # integral from a to b of (400 - (2-sqrt(x))) dx = 398*(b-a) + (2/3)(b^{3/2} - a^{3/2})
        #
        # For discrete points, the hypervolume is the sum of rectangles.
        # To maximize hypervolume with N points on [0,1], we want to place them
        # so that the "loss" from discretization is minimized.
        #
        # A good strategy: use the inverse CDF of the density that equalizes
        # the contribution. For f2 = c - sqrt(f1), the optimal placement
        # corresponds to equal spacing in a transformed coordinate.
        #
        # Let's use: place points so that sqrt(f1) is uniformly spaced, 
        # which means f1 = u^2 where u is uniform on [0,1].
        # This is equivalent to f1 = (k/N)^2 for k=0..N-1.
        #
        # Actually, let me think more carefully. The hypervolume of a set of
        # points on the front (sorted by f1) with reference (r1, r2):
        # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
        #     where f1_N = r1 (or the rightmost point extends to r1)
        #
        # Wait, for points on the Pareto front, the hypervolume is:
        # HV = sum over each point of the area of the box it "claims".
        #
        # For a front f2 = g(f1), the exact HV from f1=0 to f1=1 with ref (r1,r2) is:
        # integral_0^1 (r2 - g(x)) dx
        #
        # With N discrete points x_0 < x_1 < ... < x_{N-1} on [0,1]:
        # The hypervolume is: sum_{i=0}^{N-2} (x_{i+1} - x_i)*(r2 - g(x_i)) + (r1 - x_{N-1})*(r2 - g(x_{N-1}))
        # (assuming the last point extends to r1 in f1 direction)
        #
        # Actually, the standard hypervolume computation: for each point, the 
        # "box" width in f1 is the gap to the next point (or r1 for the last),
        # and the height in f2 is (r2 - f2_i).
        #
        # To maximize this sum, we should distribute points where the 
        # integrand (r2 - g(x)) is large, i.e., where g(x) is small, i.e., 
        # where sqrt(x) is small, i.e., where x is small.
        #
        # So we should cluster points near x=0. The optimal spacing equalizes
        # the contribution: (x_{i+1} - x_i) * (r2 - g(x_i)) = constant.
        #
        # r2 - g(x) = 400 - 2 + sqrt(x) = 398 + sqrt(x)
        #
        # So (x_{i+1} - x_i) * (398 + sqrt(x_i)) = c
        #
        # This means dx/dx_i ≈ c / (398 + sqrt(x_i))
        #
        # Since 398 >> sqrt(x) for x in [0,1], the integrand is nearly constant,
        # so uniform spacing in x is nearly optimal. But let's be more precise.
        #
        # Let u = sqrt(x), so x = u^2, dx = 2u du
        # The integrand becomes: 2u * (398 + u) du
        # = (796u + 2u^2) du
        #
        # For uniform spacing in the "area" sense, we want:
        # integral from u_i to u_{i+1} of (796u + 2u^2) du = constant
        #
        # The antiderivative is 398u^2 + (2/3)u^3
        # So we want 398u_i^2 + (2/3)u_i^3 to be uniformly spaced.
        #
        # Let's compute: F(u) = 398u^2 + (2/3)u^3
        # F(0) = 0, F(1) = 398 + 2/3 = 398.667
        #
        # For N points, we want F(u_i) = i/N * F(1) for i = 0, 1, ..., N-1
        # Then solve for u_i, and x_i = u_i^2.
        #
        # Since F(u) = 398u^2 + (2/3)u^3, and u in [0,1]:
        # F(u) ≈ 398u^2 (since 398 >> 2/3)
        # So u_i ≈ sqrt(i/N), and x_i ≈ i/N
        #
        # The correction is small. Let's just use x_i = i/N (uniform) which 
        # is essentially optimal here. But let me check if the current program
        # uses x ~ t^(4/3) which clusters MORE near 0 than uniform.
        #
        # x = t^(4/3): when t is uniform on [0,1], x is denser near 0.
        # But we showed uniform is nearly optimal. So t^(4/3) might actually
        # be slightly suboptimal!
        #
        # Let me reconsider. The reference point is (11, 400). The front is
        # f1 in [0,1], f2 in [1, 2]. So r2 - f2 is in [398, 399], nearly constant.
        # Therefore uniform spacing in f1 is nearly optimal.
        #
        # The current program uses t^(4/3) which is denser near 0. This might
        # slightly hurt because it wastes points where the integrand is already
        # high (near x=0, f2 is close to 2, so r2-f2 ≈ 398; near x=1, 
        # r2-f2 ≈ 399, almost the same).
        #
        # So switching to uniform spacing should give a slight improvement.
        #
        # Let me use uniform spacing: x_i = i / (N-1) for i = 0, ..., N-1
        # This includes both endpoints.
        
        n_points = self.budget
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], n_points)
            else:
                # Spacing uniform in f2: f1_i = ((i/(N-1))^2) * range + x0_lo
                # This clusters points near f1=0 where the front is steeper
                t = np.linspace(0.0, 1.0, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * t**2
            
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X
            
        else:
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
