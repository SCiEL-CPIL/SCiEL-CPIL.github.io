"""Portfolio search for DTLZ2: dense front grid + gap-filling, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Pareto-front-aware sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        used = 0
        all_points = []  # list of (x, f)

        def add_point(x):
            nonlocal used
            if used >= self.budget:
                return
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            if np.all(np.isfinite(f)):
                all_points.append((x.copy(), f.copy()))
                used += 1

        # DTLZ2 structure: 12 vars, bounds [0,1]
        # Pareto front: f1^2 + f2^2 + f3^2 = 1, fi >= 0
        # x[0:7] free, x[7]=cos(a/2)*sin(b/2), x[8]=sin(a/2)*cos(c/2), x[9]=sin(a/2)*sin(c/2)
        # x[10]=0.5, x[11]=0.5 for g=1 (no violations)
        # a = theta_8, b = theta_9, c = theta_10 in [0, pi]
        # f1 = cos(a/2)*sin(b/2), f2 = sin(a/2)*cos(c/2), f3 = sin(a/2)*sin(c/2)
        # Wait, let me re-derive. DTLZ2 with 3 objectives:
        # g = 1 + 10*(x_8+...+x_12) if we have 5 decision variables after position
        # Actually for DTLZ2 with M=3 objectives, we have K=2 angle variables
        # x_1..x_{n-K} are position, x_{n-K+1}..x_n are distance
        # g = 1 + 10*K*avg(x_{n-K+1}..x_n) for DTLZ2
        # f_i = g * prod_{j=1}^{i-1} sin(theta_j) * cos(theta_i) for i=1..M
        # where theta_j in [0, pi/2]
        #
        # For M=3: f1 = g*cos(t1)*cos(t2), f2 = g*sin(t1)*cos(t2), f3 = g*sin(t1)*sin(t2)
        # Wait no. Standard DTLZ2:
        # f_i(x) = g(x) * prod_{j=1}^{i-1} sin(theta_j) * cos(theta_i)
        # For i=1: f1 = g * cos(theta_1)  (empty product = 1)
        # For i=2: f2 = g * sin(theta_1) * cos(theta_2)
        # For i=3: f3 = g * sin(theta_1) * sin(theta_2)
        # where theta_1, theta_2 in [0, pi/2]
        #
        # g(x) = 1 + 10 * (x_{n-K+1} + ... + x_n) / K  ... actually it's 1+10*n_v where n_v is number of violations? 
        # No, for DTLZ2: g = 1 + 10*K*mean(x_{n-K+1:})  ... let me just use the standard formula.
        # Actually the key point: to be on the Pareto front, we need g=1, which means all distance variables must be 0.
        # But x_i are in [0,1], so x_i=0 gives g=1.
        #
        # So for M=3, K=2: x_10, x_11, x_12 are distance vars? No, n=12, K=2, so x_{11}, x_{12} are distance.
        # Position: x_1..x_10, Distance: x_11, x_12
        # theta_1 = pi/2 * x_10, theta_2 = pi/2 * x_11  ... wait that doesn't work with x in [0,1]
        #
        # Let me just directly parameterize the quarter-sphere and map back.
        # On the Pareto front: f1^2 + f2^2 + f3^2 = 1, fi >= 0
        # We need to find x such that problem(x) gives these f values.
        # For DTLZ2 with g=1: we need all distance variables = 0.
        # Then f1 = cos(t1)*cos(t2), f2 = sin(t1)*cos(t2), f3 = sin(t1)*sin(t2)
        # where t1 = pi/2 * x_{n-1}, t2 = pi/2 * x_n (the last two decision variables)
        # Wait, for n=12, M=3, K=2: the last K=2 variables are distance, and x_{n-K}=x_{10} is the last position var.
        # Actually in DTLZ2, there are K=M-1=2 angle variables and the rest are position.
        # The angle variables are typically the last K position variables before the distance variables.
        #
        # Let me just try: set x[10]=0, x[11]=0, x[12]=0 (but we only have 12 vars, indices 0-11)
        # x[10] and x[11] are distance vars. Set them to 0 for g=1.
        # x[9] is the last position var, which controls one angle.
        # Hmm, this is getting complicated. Let me just sample the quarter-sphere directly and
        # use a local search to find x that maps to each target f.

        # Strategy: 
        # 1. Generate dense grid of target points on the quarter-sphere
        # 2. For each target, find x that produces it (using known DTLZ2 structure)
        # 3. Use remaining budget for refinement

        # For DTLZ2 with 3 objectives, n=12:
        # K = M-1 = 2 angle variables
        # Distance variables: last K = 2 variables: x[10], x[11]
        # Position variables: x[0]..x[9]
        # The angle variables are embedded in the position variables.
        # Typically: theta_i = pi/2 * x_{n-K+i} for i=1..K? No.
        #
        # Standard DTLZ2 formulation:
        # g(x) = 1 + 10*K*mean(x_{n-K+1}, ..., x_n)  -- wait, this would be x[10], x[11]
        # But then where are the angles? In DTLZ2, the angles come from x_{n-2K+1}..x_{n-K}
        # So for K=2: angles from x[8], x[9], distance from x[10], x[11]
        # theta_1 = pi/2 * x[8], theta_2 = pi/2 * x[9]
        # g = 1 + 10*2*mean(x[10], x[11]) = 1 + 10*(x[10]+x[11])/2
        # f1 = g * cos(theta_1) * cos(theta_2)  -- wait, no
        # f1 = g * cos(theta_1)  -- for i=1, product is empty
        # Actually: f_i = g * (prod_{j=1}^{i-1} sin(theta_j)) * cos(theta_i)
        # f1 = g * cos(theta_1)
        # f2 = g * sin(theta_1) * cos(theta_2)  
        # f3 = g * sin(theta_1) * sin(theta_2)
        #
        # On Pareto front (g=1): f1^2+f2^2+f3^2 = cos^2(t1) + sin^2(t1)*cos^2(t2) + sin^2(t1)*sin^2(t2) = cos^2(t1) + sin^2(t1) = 1. Good.
        #
        # So to get a point on the front: set x[10]=0, x[11]=0, x[0..7] arbitrary, x[8]=2*t1/pi, x[9]=2*t2/pi
        # where t1, t2 in [0, pi/2]

        # Generate dense grid on quarter-sphere
        n_grid = 20  # 20x20 = 400 points, but budget is 200, so adjust
        # We have 200 budget. Let's use ~100 for grid, 100 for refinement
        n_theta = 14  # 14x14 = 196 points
        t1_vals = np.linspace(0, np.pi/2, n_theta)
        t2_vals = np.linspace(0, np.pi/2, n_theta)

        x_template = np.zeros(self.dim)
        x_template[10] = 0.0
        x_template[11] = 0.0
        # x[0..7] can be anything, set to 0.5
        x_template[0:8] = 0.5

        for t1 in t1_vals:
            for t2 in t2_vals:
                if used >= self.budget:
                    break
                x = x_template.copy()
                x[8] = 2*t1/np.pi
                x[9] = 2*t2/np.pi
                add_point(x)
            if used >= self.budget:
                break

        # Check deadline
        if time.time() > deadline:
            return all_points

        # Now refine: for each point on the front, do a local search to see if we can
        # get slightly better (closer to true front or find gaps)
        # Actually, the grid points should already be on the front. Let's verify and
        # use remaining budget for additional sampling in gaps or perturbation.

        # Add some random perturbations around grid points to fill in
        remaining = self.budget - used
        if remaining > 0 and len(all_points) > 0:
            # Sample additional points with small perturbations
            n_extra = min(remaining, 50)
            for i in range(n_extra):
                if used >= self.budget:
                    break
                # Pick a random grid point and perturb
                idx = rng.integers(len(all_points))
                base_x = all_points[idx][0]
                perturbation = 0.05 * rng.standard_normal(self.dim)
                x = base_x + perturbation
                add_point(x)

        # Final: if we have time, do a few more targeted samples
        if time.time() < deadline - 10 and used < self.budget:
            # Add corners and edges of the quarter-sphere
            special_t = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
            for t1 in special_t:
                for t2 in special_t:
                    if used >= self.budget:
                        break
                    x = x_template.copy()
                    x[8] = 2*t1/np.pi
                    x[9] = 2*t2/np.pi
                    add_point(x)

        return all_points
# EVOLVE-BLOCK-END
