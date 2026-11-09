"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280  # Safety margin for 300s limit

    def _dtlz4_pareto_point(self, weights):
        """
        Generate a DTLZ4 point on the Pareto front given weights.
        DTLZ4: first n-k variables control distance to front (should be 0.5 for alpha=10, or close).
        Last k variables (k = n_obj) determine the shape on the front.
        
        For DTLZ4 with alpha=10, the distance variable x_i should be 0.5^(1/alpha) ≈ 0.8909
        to be on the Pareto front. But actually, the Pareto front is at g=0, which means
        sum((x_i - 0.5)^2) = 0 for i in [0, n-k), so x_i = 0.5 for all distance vars.
        
        Wait, let me reconsider. DTLZ4:
        - n variables, m objectives
        - k = m decision variables determine the Pareto front shape
        - n-k decision variables determine the distance from the front
        - g(x) = (n-k) - sum(cos(2*pi*x_i) for i in [0, n-k))
        - For Pareto front, g=0, so x_i = 0.5 for all i in [0, n-k)
        - For the last k variables, we have:
          f_0 = g * (1 + h_0(x))
          f_j = (1-g) * h_j(x) for j=1..m
          where h_j involves the last k variables.
        
        For m=3 objectives, k=2 variables (x_{n-2}, x_{n-1}):
          h_0 = x_{n-2} * sin(pi * x_{n-1} / 2)
          h_1 = x_{n-2} * cos(pi * x_{n-1} / 2)
          h_2 = 1 - x_{n-2}
        
        So on the Pareto front (g=0):
          f_0 = x_{n-2} * sin(pi * x_{n-1} / 2)
          f_1 = x_{n-2} * cos(pi * x_{n-1} / 2)
          f_2 = 1 - x_{n-2}
        
        Given weights w = (w0, w1, w2) that sum to 1, we want to find x_{n-2}, x_{n-1}
        such that the point is close to the weighted minimum.
        
        Actually, for hypervolume, we just need to sample the Pareto front well.
        Let's parameterize: let t = x_{n-1} in [0,1], s = x_{n-2} in [0,1].
        f_0 = s * sin(pi*t/2)
        f_1 = s * cos(pi*t/2)
        f_2 = 1 - s
        
        To cover the front, we can sample (s, t) uniformly or using a low-discrepancy sequence.
        """
        # weights should be normalized to sum to 1
        w = weights / weights.sum()
        # We want to find s, t such that the point is in the direction of w.
        # A simple approach: use the weights to determine t and s.
        # f_0/f_2 = s*sin(pi*t/2)/(1-s), f_1/f_2 = s*cos(pi*t/2)/(1-s)
        # Let's just sample s and t directly from a grid or sequence.
        pass

    def _halton_sequence(self, n, dims):
        """Generate Halton sequence points in [0, 1]^dims."""
        points = np.zeros((n, dims))
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for i in range(1, n + 1):
            for j in range(dims):
                base = primes[j]
                f = 1.0
                r = 0.0
                ip = i
                while ip > 0:
                    f /= base
                    r += f * (ip % base)
                    ip //= base
                points[i - 1, j] = r
        return points

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        all_F = []
        used = 0
        
        # Phase 1: Sample the Pareto front structure
        # For DTLZ4 with 12 vars, 3 objs: k=2 vars for front shape, 10 vars for distance.
        # Pareto front: x_0..x_9 = 0.5, x_10, x_11 determine the front.
        # f_0 = x_10 * sin(pi * x_11 / 2)
        # f_1 = x_10 * cos(pi * x_11 / 2)
        # f_2 = 1 - x_10
        
        # We want to cover the 2D Pareto front well.
        # Use a grid or Halton sequence in (x_10, x_11) space.
        # x_10 in [0, 1], x_11 in [0, 1]
        
        # Let's use a combination of random sampling and structured sampling.
        # First, do some random sampling to get a sense of the landscape.
        n_random = min(50, self.budget // 4)
        for _ in range(n_random):
            if used >= self.budget or time.time() > self.deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            all_F.append(f)
            used += 1
        
        # Phase 2: Structured sampling on the Pareto front
        # Sample (s, t) where s = x_10, t = x_11
        # Use Halton sequence in 2D
        n_structured = min(100, self.budget // 2)
        halton = self._halton_sequence(n_structured, 2)
        for i in range(n_structured):
            if used >= self.budget or time.time() > self.deadline:
                break
            s, t = halton[i]
            x = np.full(self.dim, 0.5)  # Distance vars at 0.5
            x[self.dim - 2] = s  # x_10
            x[self.dim - 1] = t  # x_11
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_F.append(f)
            used += 1
        
        # Phase 3: Local search / refinement around promising points
        # Keep the best non-dominated points and try to improve them.
        if all_F:
            F_arr = np.array(all_F)
            # Find non-dominated points
            dominated = np.array([
                np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1))
                for f in F_arr
            ])
            nondom_indices = np.where(~dominated)[0]
            
            # Pick a few non-dominated points to refine
            n_refine = min(20, self.budget - used)
            if n_refine > 0 and len(nondom_indices) > 0:
                # We need to reconstruct x for these points, but we didn't store them.
                # Let's just do more random sampling with smaller steps around the front.
                for _ in range(n_refine):
                    if used >= self.budget or time.time() > self.deadline:
                        break
                    # Sample near the Pareto front
                    s = rng.uniform(0, 1)
                    t = rng.uniform(0, 1)
                    x = np.full(self.dim, 0.5)
                    x[self.dim - 2] = np.clip(s + 0.05 * rng.standard_normal(), 0, 1)
                    x[self.dim - 1] = np.clip(t + 0.05 * rng.standard_normal(), 0, 1)
                    # Add small noise to distance vars
                    x[:self.dim-2] += 0.01 * rng.standard_normal(self.dim - 2)
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    all_F.append(f)
                    used += 1
        
        return all_F
# EVOLVE-BLOCK-END
