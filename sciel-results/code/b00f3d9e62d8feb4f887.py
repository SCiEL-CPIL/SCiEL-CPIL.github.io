# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for WFG7 using LHS global sampling + local Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit with safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim

        # Phase 1: Latin Hypercube Sampling for diverse global coverage
        # Use about 60% of budget for global exploration
        n_lhs = int(self.budget * 0.6)
        n_lhs = max(n_lhs, 30)  # minimum reasonable sample

        sampler = qmc.LatinHypercube(d=dim, seed=42)
        lhs_samples = sampler.random(n=n_lhs)
        # Map [0,1] to [lo, hi]
        lhs_x = lo + lhs_samples * (hi - lo)

        # Evaluate all LHS points
        archive = []  # list of (x, f) tuples
        eval_count = 0

        for i in range(n_lhs):
            if time.time() > deadline - 30:
                break
            x = lhs_x[i]
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            archive.append((x, f))

        # Phase 2: Local refinement using Nelder-Mead on promising points
        # Select top non-dominated points and refine them
        if eval_count < self.budget:
            # Find non-dominated points in the archive
            non_dominated = self._get_non_dominated(archive)

            # Sort by sum of objectives (a simple proxy for quality)
            non_dominated.sort(key=lambda xf: np.sum(xf[1]))

            # Refine the best few points locally
            n_refine = min(len(non_dominated), 5)
            for i in range(n_refine):
                if time.time() > deadline - 15:
                    break
                if eval_count >= self.budget:
                    break

                x0, f0 = non_dominated[i]

                # Define scalarized objective for local optimization
                # Use weighted sum with weights based on ideal point
                def scalarized(x):
                    fx = np.array(problem(x), dtype=np.float64)
                    # Simple weighted sum
                    return np.sum(fx)

                # Try Nelder-Mead from this starting point
                try:
                    result = minimize(
                        scalarized,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    eval_count += result.nfev
                    if result.fun < np.sum(f0):
                        archive.append((result.x, np.array(problem(result.x), dtype=np.float64)))
                        eval_count += 1
                except Exception:
                    pass

                # Check time and budget
                if time.time() > deadline - 10:
                    break

        # Return the best point found (by hypervolume contribution proxy)
        # The evaluator records all objective vectors from problem calls,
        # so we just need to ensure we've made good evaluations.
        # Return something reasonable
        if archive:
            # Return the point with best scalarized value
            best = min(archive, key=lambda xf: np.sum(xf[1]))
            return best[0]
        else:
            return (lo + hi) / 2

    def _get_non_dominated(self, archive):
        """Return list of non-dominated (x, f) tuples from archive."""
        if not archive:
            return []

        n = len(archive)
        dominated = [False] * n

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                fi = archive[i][1]
                fj = archive[j][1]
                # Check if i dominates j
                if np.all(fi <= fj) and np.any(fi < fj):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(fj <= fi) and np.any(fj < fi):
                    dominated[i] = True
                    break

        return [archive[i] for i in range(n) if not dominated[i]]
# EVOLVE-BLOCK-END
