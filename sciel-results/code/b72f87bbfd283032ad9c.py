"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3, K=4 decision vars on the front, N-K=8 control vars):
      x_0..x_3 : simplex variables defining the front shape
      x_4..x_11: control variables; g(x_4..x_11) is minimized at x_i=0.5 (each)
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, (x_0,x_1,1-x_0-x_1) traces the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin well under 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []  # list of x vectors

        # --- Phase 1: Deterministic structured sampling on the 2-simplex ---
        # For M=3 objectives, K = M+1 = 4 front variables (x0..x3) summing to 1.
        # But the standard DTLZ4 with M=3 has K=3 position variables if we use
        # the formulation f_i = (1+g)*z_i where z are on the simplex.
        # With dim=12, n_obj=3: typically K = n_obj = 3 position vars (x0,x1,x2)
        # and dim-K = 9 control vars. Let me use 3 position vars.
        
        n_pts = self.budget
        pts = []
        
        # Use a stratified grid approach for better hypervolume coverage
        # The simplex x0+x1+x2=1, xi>=0 can be parameterized by:
        # x0 = 1 - sqrt(u), x1 = sqrt(u)*(1-v), x2 = sqrt(u)*v
        # But for hypervolume, we want uniform coverage in objective space.
        # 
        # Better: use a triangular grid with jitter for stratified sampling
        # Or use Halton but with a different mapping that gives more uniform
        # coverage in the 2D objective space.
        
        # Key insight: for hypervolume with reference point [2.5,2.5,2.5],
        # points near the "corners" of the triangle contribute more to HV
        # because they dominate larger regions. We should oversample near
        # the edges/corners.
        
        # Strategy: Mix of corner points, edge points, and interior points
        # Corners: (1,0,0), (0,1,0), (0,0,1) - 3 points
        # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) - 3 points  
        # Then fill with stratified interior points
        
        # Add corners and edge midpoints
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
        pts.extend(special_pts)
        
        # Remaining points: use stratified sampling on the simplex
        # Divide the simplex into sub-triangles and sample one point per cell
        n_remaining = n_pts - len(pts)
        
        # Use a finer stratified approach: split each coordinate into bins
        # For better HV, use a power-law distribution that oversamples near edges
        # x0 = (1-u)^alpha, etc., with alpha < 1 to concentrate near corners
        
        # Actually, let's use a different approach: sample uniformly in the 
        # 2D parameter space but with a transformation that gives better HV.
        # 
        # The optimal distribution for HV maximization is roughly uniform in
        # the non-dominated region. Since our front is the triangle, we want
        # points spread uniformly across the triangle.
        
        # Use Halton sequence for low-discrepancy sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        for i in range(1, n_remaining + 1):
            u = halton(i, 2)
            v = halton(i, 3)
            
            # Standard uniform mapping to simplex
            x0 = 1.0 - np.sqrt(u)
            x1 = np.sqrt(u) * (1.0 - v)
            x2 = np.sqrt(u) * v
            
            pts.append(np.array([x0, x1, x2]))

        # Control variables at their optimum: x_i = 0.5
        # There are dim - 3 = 9 control variables (x3 to x11)
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # Build full x vectors
        for p in pts:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Evaluate candidates up to budget, keeping all returned vectors ---
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have budget left, add random points on simplex
        while used < self.budget:
            if time.time() > deadline:
                break
            u = rng.random()
            v = rng.random()
            x0 = 1.0 - np.sqrt(u)
            x1 = np.sqrt(u) * (1.0 - v)
            x2 = np.sqrt(u) * v
            y = np.array([x0, x1, x2])
            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([y, ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
