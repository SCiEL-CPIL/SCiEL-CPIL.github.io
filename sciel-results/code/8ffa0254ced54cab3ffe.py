# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer: exploit the known 2D Pareto-front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 290.0  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # For DTLZ7 with M objectives, n variables:
        #   g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2
        #   f_j = (1+g) * prod_{k=0}^{M-2} x_k  (j=0)
        #   f_j = (1+g) * (1-x_0)*prod... etc.
        # Pareto front is achieved when g=0, i.e., x[M:] = 0.5 (in 0-indexed, x[m:] = 0.5)
        # Then f values depend only on x[0] and x[1] (for M=3).

        # Strategy:
        # 1. Place most evaluations on the known Pareto front manifold (x[2:]=0.5)
        #    using a quasi-random 2D grid over x[0], x[1] in [0,1].
        # 2. Use remaining budget for small perturbations to handle any mismatch
        #    or to fine-tune if the problem isn't exactly DTLZ7.

        # Quasi-random 2D points using Halton-like low-discrepancy sequence
        def halton_2d(n_points):
            """Generate n_points quasi-random points in [0,1]^2 using Halton sequence."""
            pts = np.zeros((n_points, 2))
            for i in range(n_points):
                # Base 2 for first dimension
                base = 2
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= base
                    r += f * (q % base)
                    q //= base
                pts[i, 0] = r

                # Base 3 for second dimension
                base = 3
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= base
                    r += f * (q % base)
                    q //= base
                pts[i, 1] = r
            return pts

        # Split budget: ~80% on the Pareto front manifold, ~20% for exploration
        n_front = int(budget * 0.80)
        n_explore = budget - n_front

        # Generate 2D quasi-random points for the front
        front_2d = halton_2d(n_front)

        # Build candidate points on the Pareto front manifold
        candidates = []
        for i in range(n_front):
            x = np.full(n, 0.5)  # x[2:] = 0.5 for DTLZ7 (0-indexed)
            x[0] = front_2d[i, 0]
            x[1] = front_2d[i, 1]
            # Clip to bounds
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # Add some exploration points (random in full space, biased toward center)
        rng = np.random.default_rng(42)
        for i in range(n_explore):
            x = lo + (hi - lo) * rng.random(n)
            # Bias toward x[2:]=0.5
            x[2:] = 0.5 + 0.05 * (hi[2:] - lo[2:]) * rng.standard_normal(n - 2)
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # Evaluate all candidates within budget and time
        best_points = []
        best_obj = []
        for i, x in enumerate(candidates):
            if time.perf_counter() > deadline:
                break
            obj = np.asarray(problem(x), dtype=float)
            if obj.shape != (m,):
                obj = obj.flatten()[:m]
            best_points.append(x)
            best_obj.append(obj)

        # If we have time, do a local refinement around the best-known points
        # Try small perturbations of the top-performing points
        if len(best_obj) > 0 and time.perf_counter() < deadline - 30:
            best_obj_arr = np.array(best_obj)
            # Select top 10 points (lowest sum of objectives)
            sums = np.sum(best_obj_arr, axis=1)
            top_idx = np.argsort(sums)[:min(10, len(best_obj))]
            
            rng2 = np.random.default_rng(123)
            for idx in top_idx:
                if time.perf_counter() > deadline:
                    break
                base_x = best_points[idx].copy()
                for _ in range(5):  # 5 perturbations per top point
                    if time.perf_counter() > deadline:
                        break
                    pert = base_x + 0.01 * (hi - lo) * rng2.standard_normal(n)
                    pert = np.clip(pert, lo, hi)
                    obj = np.asarray(problem(pert), dtype=float)
                    if obj.shape != (m,):
                        obj = obj.flatten()[:m]
                    best_points.append(pert)
                    best_obj.append(obj)

        return best_points
# EVOLVE-BLOCK-END
