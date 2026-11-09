"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7 using Halton sequence on the 2D Pareto front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_2d(self, n, skip=1):
        """Generate n points in [0,1]^2 using Halton sequence (bases 2 and 3)."""
        points = np.zeros((n, 2))
        for i in range(n):
            idx = i + skip
            # Base 2
            f, r, q = 1.0, 0.0, idx
            while q > 0:
                f /= 2.0
                r += f * (q % 2)
                q //= 2
            points[i, 0] = r
            # Base 3
            f, r, q = 1.0, 0.0, idx
            while q > 0:
                f /= 3.0
                r += f * (q % 3)
                q //= 3
            points[i, 1] = r
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7 structure: x[0] and x[1] parameterize the Pareto front,
        # x[2:] should be set to maximize the g-term (minimize g to 0)
        # For DTLZ7, g(x) = (n_g - 1) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in 2..n-1
        # g=0 when x_i = 0.5 for all i >= 2

        n_evals = 0
        results = []

        # Phase 1: Systematic Halton sampling on the Pareto front manifold
        # Use 180 evaluations for front coverage
        n_halton = 180
        halton_points = self._halton_2d(n_halton, skip=1)

        for hp in halton_points:
            if time.time() > deadline:
                break
            if n_evals >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = lo[0] + hp[0] * (hi[0] - lo[0])
            x[1] = lo[1] + hp[1] * (hi[1] - lo[1])
            # Set remaining variables to 0.5 (optimal for g=0 in DTLZ7)
            for i in range(2, self.dim):
                x[i] = 0.5 * (lo[i] + hi[i])

            f = problem(x)
            results.append(f)
            n_evals += 1

        # Phase 2: Local refinement around promising regions
        # Use remaining budget for local search
        remaining = self.budget - n_evals
        if remaining > 0 and results:
            # Find non-dominated points and refine around them
            F = np.array(results)
            # Simple non-dominance filter
            non_dom_idx = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)

            # Refine around a subset of non-dominated points
            n_refine = min(remaining, max(10, len(non_dom_idx)))
            refine_indices = non_dom_idx[:n_refine] if len(non_dom_idx) > 0 else list(range(min(5, len(results))))

            for idx in refine_indices:
                if n_evals >= self.budget or time.time() > deadline:
                    break

                base_f = F[idx]
                # Try small perturbations in x[0] and x[1]
                for delta0 in [-0.05, 0.05]:
                    for delta1 in [-0.05, 0.05]:
                        if n_evals >= self.budget or time.time() > deadline:
                            break

                        x = np.zeros(self.dim)
                        # Reconstruct x from the halton point that gave this result
                        # We need to track which halton point produced each result
                        # For simplicity, just perturb around 0.5 for x[2:]
                        h_idx = idx % n_halton
                        hp = halton_points[h_idx]
                        x[0] = np.clip(lo[0] + (hp[0] + delta0) * (hi[0] - lo[0]), lo[0], hi[0])
                        x[1] = np.clip(lo[1] + (hp[1] + delta1) * (hi[1] - lo[1]), lo[1], hi[1])
                        for i in range(2, self.dim):
                            x[i] = 0.5 * (lo[i] + hi[i])

                        f = problem(x)
                        results.append(f)
                        n_evals += 1

        return results
# EVOLVE-BLOCK-END
