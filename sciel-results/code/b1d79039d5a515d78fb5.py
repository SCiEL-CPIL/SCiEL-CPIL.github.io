"""Multi-objective optimizer on RE21 using Sobol sampling + Nelder-Mead local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer: Sobol init + Nelder-Mead polish on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def add_to_archive(x, f):
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Sobol sequence initialization (48 points for good coverage)
        from scipy.stats import qmc
        n_init = 48
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sobol_samples = sampler.random(n_init)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + (hi[d] - lo[d]) * sobol_samples[i, d]
            evaluate(x)

        # Phase 2: Nelder-Mead local optimization with multiple weight vectors
        if time.time() < deadline - 30.0 and len(archive) > 0:
            from scipy.optimize import minimize

            # Use a set of weight vectors to cover the Pareto front
            # For 2 objectives, use evenly spaced weights
            n_weights = 8
            weights = [(1.0 - (i + 0.5) / n_weights, (i + 0.5) / n_weights) for i in range(n_weights)]
            # Add extreme weights
            weights.append((0.95, 0.05))
            weights.append((0.05, 0.95))

            # Get initial points from archive (spread across the front)
            # Sort by first objective to get diverse starting points
            sorted_archive = sorted(archive, key=lambda xf: xf[1][0])
            n_starts = min(len(sorted_archive), 12)
            start_points = sorted_archive[::max(1, len(sorted_archive) // n_starts)][:n_starts]
            if len(start_points) < n_starts:
                start_points = sorted_archive

            for x_init, f_init in start_points:
                if used >= self.budget or time.time() > deadline - 20.0:
                    break

                for w in weights:
                    if used >= self.budget or time.time() > deadline - 15.0:
                        break

                    def scalar_obj(x, w=w):
                        f = problem(x)
                        return w[0] * f[0] + w[1] * f[1]

                    try:
                        result = minimize(
                            scalar_obj,
                            x_init,
                            method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True}
                        )
                        if used < self.budget and time.time() < deadline - 10.0:
                            f_new = problem(result.x)
                            used += 1
                            if is_valid(f_new):
                                add_to_archive(result.x, f_new)
                    except Exception:
                        pass

        # Phase 3: Final tight polish on top archive points
        if time.time() < deadline - 10.0 and len(archive) > 0 and used < self.budget:
            from scipy.optimize import minimize

            # Take the 5 best points (by hypervolume contribution approximation)
            archive_sorted = sorted(archive, key=lambda xf: xf[1][0] + xf[1][1])
            top_points = archive_sorted[:5]

            for x_init, f_init in top_points:
                if used >= self.budget or time.time() > deadline - 5.0:
                    break

                # Try a few weight vectors for final polish
                for w in [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]:
                    if used >= self.budget or time.time() > deadline - 5.0:
                        break

                    def scalar_obj(x, w=w):
                        f = problem(x)
                        return w[0] * f[0] + w[1] * f[1]

                    try:
                        result = minimize(
                            scalar_obj,
                            x_init,
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'adaptive': True}
                        )
                        if used < self.budget and time.time() < deadline - 3.0:
                            f_new = problem(result.x)
                            used += 1
                            if is_valid(f_new):
                                add_to_archive(result.x, f_new)
                    except Exception:
                        pass

        return archive

# EVOLVE-BLOCK-END
