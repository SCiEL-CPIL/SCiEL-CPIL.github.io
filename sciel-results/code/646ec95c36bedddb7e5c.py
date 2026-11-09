"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage global-to-local search: Sobol sampling then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Stage 1: Sobol quasi-random sampling
        n_samples = min(400, budget // 2)
        # Generate Sobol sequence
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_samples = sampler.random(n_samples)
        except ImportError:
            # Fallback to random if scipy not available
            sobol_samples = rng.random((n_samples, dim))

        # Map to [-5, 5]
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Evaluate all samples
        values = np.empty(n_samples)
        for i in range(n_samples):
            x = sobol_samples[i]
            values[i] = func(x)
            if time.time() > deadline - 60:
                break

        # Get top 5 candidates
        n_top = min(5, n_samples)
        top_indices = np.argsort(values)[:n_top]
        top_points = sobol_samples[top_indices]
        top_values = values[top_indices]

        # Stage 2: Nelder-Mead local optimization from top candidates
        best_x = top_points[0]
        best_f = top_values[0]
        remaining_budget = budget - n_samples

        # Distribute remaining budget among top candidates
        calls_per_candidate = max(10, remaining_budget // n_top)

        for k in range(n_top):
            if time.time() > deadline - 30:
                break

            x0 = top_points[k].copy()
            f0 = top_values[k]

            # Adaptive initial simplex size
            if k == 0:
                simplex_size = 1.0
            else:
                # Scale by distance to best point
                dist = np.linalg.norm(x0 - best_x)
                simplex_size = max(0.5, min(3.0, dist * 0.5))

            # Implement Nelder-Mead manually to control budget
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]

            for i in range(dim):
                perturbation = x0.copy()
                if abs(perturbation[i]) > 0.01:
                    perturbation[i] *= 1.05
                else:
                    perturbation[i] += 0.05 * simplex_size
                simplex.append(perturbation)
                simplex_f.append(func(perturbation))

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            nm_calls = 0
            max_nm_iters = calls_per_candidate

            for _ in range(max_nm_iters):
                if time.time() > deadline - 20:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                nm_calls += 1

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    nm_calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        nm_calls += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            break
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        nm_calls += 1
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(simplex[i])
                                nm_calls += 1
                            if nm_calls > max_nm_iters:
                                break

                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

        # Final check
        if time.time() > deadline:
            return best_x, best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
