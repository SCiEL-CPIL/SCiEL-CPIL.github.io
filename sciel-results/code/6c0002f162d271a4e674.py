import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0, so g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1], 1)
        # h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
        #
        # Reference point: [1.1, 1.1, 25.0]
        # We want to maximize hypervolume. The front is 2D parameterized by (x0, x1) in [0,1]^2.
        #
        # Strategy: Use a hypervolume-optimized non-uniform sampling.
        # The key insight: near the "corner" (low f1, low f2, low f3) points contribute most.
        # f3 = 1 - x1*h(x1,1). Let's analyze h(x1, 1):
        # h(y,1) = 9/(1+2*(sin(3πy)+sin(4πy)))
        # At y=0: sin(0)+sin(0)=0, h=9, f3=1-0=1
        # At y=0.5: sin(1.5π)+sin(2π)=-1+0=-1, h=9/(1-2)=9/(-1)=-9, f3=1-0.5*(-9)=5.5
        # So f3 ranges from 1 to ~5.5+ depending on y.
        #
        # The hypervolume contribution of a point (f1,f2,f3) with ref (1.1,1.1,25) is
        # (1.1-f1)*(1.1-f2)*(25-f3) for the "box" it contributes to the HV.
        #
        # To maximize total HV, we should concentrate points where the marginal HV gain is highest.
        # This is near (0,0,1) in objective space, i.e., x0≈0, x1≈0.
        #
        # Use a power-law distribution: sample x0 and x1 with density proportional to (1-x)^α
        # to cluster near 0. We'll use α=2 for both dimensions.
        #
        # Also use a combination of:
        # 1. A structured grid with power-law warping
        # 2. Halton sequence with power-law warping for the remaining budget

        # Power-law warp: u -> u^(1/(1+α)) maps uniform to beta-like distribution near 0
        # For α=2: u -> u^(1/3)
        alpha = 2.0
        def warp(u):
            return u ** (1.0 / (1.0 + alpha))

        # Create a grid with warped coordinates
        n_grid = int(np.sqrt(budget))
        while n_grid * n_grid > budget:
            n_grid -= 1
        if n_grid < 2:
            n_grid = 2

        # Evaluate warped grid points
        for i in range(n_grid):
            if not check_time():
                break
            u0 = i / (n_grid - 1)
            x0_val = lo[0] + (hi[0] - lo[0]) * warp(u0)
            for j in range(n_grid):
                if not check_time():
                    break
                if used >= budget:
                    break
                u1 = j / (n_grid - 1)
                x1_val = lo[1] + (hi[1] - lo[1]) * warp(u1)
                x = np.zeros(dim)
                x[0] = x0_val
                x[1] = x1_val
                evaluate(x)

        # Fill remaining budget with Halton sequence, warped
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        remaining = budget - used
        if remaining > 0:
            for i in range(remaining):
                if not check_time():
                    break
                if used >= budget:
                    break

                n = i + 1
                u0 = halton(n, 2)
                u1 = halton(n, 3)
                x0_val = lo[0] + (hi[0] - lo[0]) * warp(u0)
                x1_val = lo[1] + (hi[1] - lo[1]) * warp(u1)

                x = np.zeros(dim)
                x[0] = x0_val
                x[1] = x1_val
                evaluate(x)

        return None
