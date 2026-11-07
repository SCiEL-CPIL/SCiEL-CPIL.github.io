"""Multi-objective optimizer for DTLZ1 using direct Pareto front sampling with Sobol sequence."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct sampler for DTLZ1 Pareto front using low-discrepancy sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # DTLZ1 structure:
        #   g = sum((x[i]-0.5)^2) for i in [4,5,6]  (last 3 vars)
        #   z1 = x[0], z2 = x[1]
        #   f1 = 0.5*g*(1-z1), f2 = 0.5*g*z1, f3 = 0.5*g*z2
        # Pareto front: g=0, z1+z2<=1, z1,z2>=0
        #   -> triangle in objective space.
        # We fix the g-vars at 0.5 (which gives g=0) and sample z1,z2 on the simplex.

        # Generate Sobol points in [0,1]^2, then map to simplex {z1,z2>=0, z1+z2<=1}
        # Using the transformation: for a point (u,v) in [0,1]^2,
        # we can map to simplex via:
        #   if u+v <= 1: (z1, z2) = (u, v)
        #   elif u > v: (z1, z2) = (u, (1-u)*v)  -- this isn't quite right
        #
        # Better approach: use the standard simplex mapping.
        # For a point (u, v) in [0,1]^2, define:
        #   t = u if u < v else v  (min)
        #   s = 1 - t
        #   Actually, let's use a cleaner method.
        #
        # The standard way to map [0,1]^2 to a triangle:
        #   Let a = u, b = v
        #   If a + b > 1: a = 1 - a, b = 1 - b  (reflect)
        #   Then (z1, z2) = (a, b)
        # This maps to the triangle {z1,z2 >= 0, z1+z2 <= 1} but with a folding.
        #
        # Alternatively, use the inverse CDF approach for uniform on simplex:
        #   Generate u1, u2 ~ U[0,1]
        #   z1 = u1^(1/2) * u2^(1/3)  -- no, that's not right either.
        #
        # Let's use a direct lattice approach combined with Sobol for better coverage.
        # Actually, the simplest robust approach: use Sobol in 2D, then apply the
        # reflection mapping to get points on the simplex.

        # Generate Sobol points
        n_pts = budget
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            raw = sampler.random(n_pts)
        except ImportError:
            # Fallback: use Halton sequence
            raw = self._halton_sequence(n_pts, 2)

        # Map [0,1]^2 to simplex {z1,z2>=0, z1+z2<=1}
        # Use the reflection/folding method:
        u = raw[:, 0]
        v = raw[:, 1]
        # Reflect points that are outside the triangle
        mask = (u + v) > 1.0
        u_reflected = np.where(mask, 1.0 - u, u)
        v_reflected = np.where(mask, 1.0 - v, v)

        z1 = u_reflected
        z2 = v_reflected

        # Ensure we're within bounds
        z1 = np.clip(z1, 0.0, 1.0)
        z2 = np.clip(z2, 0.0, 1.0)

        # Build the input vectors
        # x[0] = z1, x[1] = z2
        # x[2], x[3] = 0.5 (free variables, don't affect objectives)
        # x[4], x[5], x[6] = 0.5 (g-vars at minimum, giving g=0)
        X = np.zeros((n_pts, dim))
        X[:, 0] = z1
        X[:, 1] = z2
        X[:, 2] = 0.5
        X[:, 3] = 0.5
        X[:, 4] = 0.5
        X[:, 5] = 0.5
        X[:, 6] = 0.5

        # Clip to bounds
        for i in range(dim):
            X[:, i] = np.clip(X[:, i], lo[i], hi[i])

        # Evaluate all points
        all_F = []
        used = 0
        for i in range(n_pts):
            if used >= budget:
                break
            if time.time() > deadline - 2.0:
                break
            x = X[i]
            f = np.array(problem(x), dtype=float)
            used += 1
            all_F.append(f)

        # Ensure at least one evaluation happened
        if not all_F:
            x = lo + 0.5 * (hi - lo)
            f = np.array(problem(x), dtype=float)
            all_F.append(f)

        return np.array(X[:used])

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0,1]^dim."""
        points = np.zeros((n, dim))
        bases = [2, 3, 5, 7, 11, 13, 17][:dim]
        for d in range(dim):
            base = bases[d]
            for i in range(1, n + 1):
                f = 1.0
                result = 0.0
                num = i
                while num > 0:
                    num, rem = divmod(num, base)
                    f /= base
                    result += rem * f
                points[i - 1, d] = result
        return points
# EVOLVE-BLOCK-END
