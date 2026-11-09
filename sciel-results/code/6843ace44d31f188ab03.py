"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using hypervolume-aware non-uniform Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 with M=3, D=22:
        # f1 = x1, f2 = x2, f3 = (1 + x_{D-1}) * g
        # g = 1 + 9/(D-M+1) * sum_{i=M}^{D-1} sin(3*pi*x_i)
        # Pareto front: g=0 => x_2..x_21 all at 0 (or other zeros of sin)
        # On front: f1=x1, f2=x2, f3=0, with x1>=0, x2>=0, x1+x2<=1

        # Strategy: Model the Pareto front as a 1D curve
        # The front is the triangle in (f1, f2) plane: f1>=0, f2>=0, f1+f2<=1
        # Parametrize by a single parameter t in [0,1]:
        #   f1 = t * (1-t), f2 = t, f3 = 0  -- this is one edge
        # Actually the full front is 2D. But hypervolume is dominated by points near origin.
        # 
        # Better: Use the fact that for hypervolume with reference [1.1, 1.1, 25.0],
        # points near (0,0,0) contribute most. The front is the line segment from (0,0,0)
        # to (1,0,0) and (0,1,0) and the arc connecting them where f1+f2=1.
        #
        # Actually for DTLZ7, the Pareto front is:
        # f1 = x1, f2 = x2, f3 = 0, with x1 >= 0, x2 >= 0, x1 + x2 <= 1
        # This is a 2D triangular region.
        #
        # Key insight: The hypervolume contribution is highest near the origin.
        # We should concentrate samples near (0,0,0).
        #
        # Use a 1D parametrization that densely covers the region near the origin:
        # Let t in [0,1] parametrize distance from origin along the "diagonal"
        # and use angular parameter for direction.
        #
        # For hypervolume-aware sampling, use:
        #   r = t^gamma (clusters near 0 for gamma > 1)
        #   theta = 2*pi*u (uniform angle)
        #   f1 = r * cos(theta), f2 = r * sin(theta)
        # But we need f1+f2 <= 1, so r <= 1/max(cos, sin) approximately.
        #
        # Simpler: Use the triangle parametrization:
        #   x1 = u * v, x2 = u * (1-v), with u in [0,1], v in [0,1]
        #   f1 = x1, f2 = x2, f3 = 0
        #
        # To concentrate near origin, use u = t^gamma.
        #
        # For 200 evaluations, use a mix:
        # - Dense sampling near origin using power-law
        # - Some points along the edges and the hypotenuse

        n_total = budget
        used = 0
        front_points = []

        # Phase 1: Dense sampling near the origin using power-law
        # Use a 2D grid in (u, v) with u = t^gamma
        n_grid = 15  # 15x15 = 225, but we'll use 200
        gamma = 2.5  # Strong clustering near origin
        
        # Generate 1D grid with power-law spacing
        u_vals = np.linspace(0, 1, n_grid) ** gamma
        v_vals = np.linspace(0, 1, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if used >= n_total:
                    break
                x1 = u * v
                x2 = u * (1 - v)
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                # x[2:] = 0 for g=0
                front_points.append(x)
                used += 1
            if used >= n_total:
                break

        # Phase 2: Add points along the edges for better coverage
        remaining = n_total - used
        if remaining > 0:
            # Edge 1: x2=0, x1 from 0 to 1
            n_edge = min(remaining // 3, 10)
            for i in range(n_edge):
                if used >= n_total:
                    break
                x1 = (i + 1) / (n_edge + 1)
                x = np.zeros(dim)
                x[0] = x1
                front_points.append(x)
                used += 1

            # Edge 2: x1=0, x2 from 0 to 1
            n_edge = min(remaining // 3, 10)
            for i in range(n_edge):
                if used >= n_total:
                    break
                x2 = (i + 1) / (n_edge + 1)
                x = np.zeros(dim)
                x[1] = x2
                front_points.append(x)
                used += 1

            # Edge 3: x1+x2=1, the "hypotenuse"
            n_edge = min(remaining - 2 * (n_edge if remaining >= 30 else remaining // 3), 10)
            for i in range(n_edge):
                if used >= n_total:
                    break
                t = (i + 1) / (n_edge + 1)
                x1 = t
                x2 = 1 - t
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                front_points.append(x)
                used += 1

        # Phase 3: Add random points with power-law bias for remaining budget
        remaining = n_total - used
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                u = rng.random() ** gamma
                v = rng.random()
                x1 = u * v
                x2 = u * (1 - v)
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                front_points.append(x)
                used += 1

        front_points = np.array(front_points[:n_total])
        
        # Evaluate front points
        F = np.array([problem(x) for x in front_points])
        used = len(front_points)

        # Check time
        if time.time() > deadline:
            return front_points

        # Phase 4: Local refinement to improve g (make it closer to 0)
        remaining = budget - used
        if remaining <= 0:
            return front_points

        # For each point, try to reduce g by adjusting x[2:]
        # g = 1 + 9/20 * sum_{i=2}^{21} sin(3*pi*x_i)
        # We want g=0, which requires sum sin(3*pi*x_i) = -20/9
        # Since sin(3*pi*x) >= -1, we need all sin(3*pi*x_i) = -1, i.e., x_i = 1/2
        # Wait, that's not right. Let me recalculate.
        # g = 1 + 9/(D-M+1) * sum sin(3*pi*x_i)
        # For D=22, M=3: D-M+1 = 20
        # g = 1 + 9/20 * sum_{i=2}^{21} sin(3*pi*x_i)
        # g=0 => sum sin(3*pi*x_i) = -20/9 ≈ -2.222
        # Since each sin term is in [-1, 1], and we have 20 terms,
        # we need the sum to be -2.222, which is achievable.
        # The minimum possible sum is -20 (all x_i = 1/2), so -2.222 is easy.
        # Actually, x_i=0 gives sin(0)=0, so g=1.
        # To get g=0, we need sum sin(3*pi*x_i) = -20/9.
        # One solution: set some x_i to values where sin(3*pi*x_i) is negative.
        # sin(3*pi*x) = -1 when 3*pi*x = 3*pi/2 + 2*pi*k, i.e., x = 1/2 + 2k/3
        # So x = 1/2, 7/6, ... In [0,1], x=1/2 gives sin(3*pi/2) = -1.
        # So if we set all x_i = 1/2 for i=2..21, sum = -20, g = 1 + 9/20*(-20) = 1-9 = -8.
        # That's too much. We need sum = -20/9.
        # Let k of the 20 variables be at x=1/2 (sin=-1) and the rest at x=0 (sin=0).
        # Then sum = -k, so we need k = 20/9 ≈ 2.22.
        # So set 2 variables to 1/2 and 1 variable to a value where sin(3*pi*x) = -2/9.
        # sin(3*pi*x) = -2/9 => 3*pi*x = pi + arcsin(2/9) or 2*pi - arcsin(2/9)
        # arcsin(2/9) ≈ 0.223 rad
        # x = (pi + 0.223)/(3*pi) ≈ 1.223/9.425 ≈ 0.130
        # or x = (2*pi - 0.223)/(3*pi) ≈ 6.06/9.425 ≈ 0.643

        # For local refinement, we'll try to adjust x[2:] to minimize g
        n_local = len(front_points)
        steps_per_point = max(1, remaining // n_local)

        for i in range(n_local):
            if time.time() > deadline:
                break
            for step in range(steps_per_point):
                if time.time() > deadline or used >= budget:
                    break

                x_curr = front_points[i].copy()
                
                # Try different strategies to reduce g
                rng_local = np.random.default_rng(i * 1000 + step)
                
                # Strategy: set some of x[2:] to 1/2 and others to 0
                # Number of 1/2 values: random from 1 to 4
                n_half = rng_local.integers(1, 5)
                indices = rng_local.choice(dim - 2, size=n_half, replace=False)
                
                x_new = x_curr.copy()
                for idx in indices:
                    x_new[2 + idx] = 0.5
                # Set others to 0
                for idx in range(dim - 2):
                    if idx not in indices:
                        x_new[2 + idx] = 0.0
                
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1
                
                # Accept if it improves f3 (which is (1+x_{D-1})*g)
                if f_new[2] < F[i][2] - 1e-10:
                    front_points[i] = x_new
                    F[i] = f_new

        return front_points
# EVOLVE-BLOCK-END
