"""Multi-start adaptive (1+1)-ES with Nelder-Mead refinement for BBOB Weierstrass (f16).

Strategy:
  Phase 1: cheap deterministic multi-start (center + randoms) to find a good basin early.
  Phase 2: adaptive (1+1)-ES on the best start for ~60% of remaining budget.
  Phase 3: Nelder-Mead local refinement on the best point found.

AOCC rewards early progress, so we front-load cheap evaluations and keep the
ES loop going for most of the budget before polishing.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with Nelder-Mead refinement for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin below the 330s kill
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        # ---- Phase 1: minimal multi-start to find a good basin early ----
        n_starts = 6
        seeds = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            seeds.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        for s in seeds:
            if time.time() > deadline and best_x is not None:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        if best_x is None:
            x0 = np.zeros(dim)
            best_f = safe_func(x0)
            best_x = x0.copy()

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # start large to explore the oscillatory landscape
        min_sigma, max_sigma = 1e-7, 8.0
        target = 1.0 / 5.0  # 1/5 success rule
        c_sigma = 0.3  # moderate adaptation speed

        remaining_budget = self.budget - call_count
        es_budget = int(remaining_budget * 0.6)

        for _ in range(es_budget):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_func(y)
            improved = fy < fx
            if improved:
                x, fx = y, fy

            # Per-step sigma adaptation
            sigma *= np.exp(c_sigma * ((1.0 if improved else 0.0) - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        # ---- Phase 3: Nelder-Mead refinement ----
        # Use remaining budget for local refinement
        if time.time() < deadline and call_count < self.budget:
            x, fx = self._nelder_mead(
                x, fx, safe_func, dim, deadline,
                max_calls=self.budget - call_count, rng=rng
            )

        return x, fx

    def _nelder_mead(self, x0, f0, func, dim, deadline, max_calls, rng):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        alpha = 0.1  # perturbation size for initial simplex
        simplex = [x0.copy()]
        simplex_f = [f0]

        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha
            xi = np.clip(xi, -5.0, 5.0)
            fi = func(xi)
            simplex.append(xi)
            simplex_f.append(fi)

        # Nelder-Mead parameters
        alpha_n = 1.0   # reflection
        gamma = 2.0     # expansion
        rho = 0.5       # contraction
        sigma_n = 0.5   # shrink

        iterations = 0
        max_iterations = max_calls - (dim + 1)  # we already used dim+1 calls

        while iterations < max_iterations:
            if time.time() > deadline:
                break

            iterations += 1

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha_n * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)

            if fr < simplex_f[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)

                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_n * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = func(simplex[i])

            # Update best
            best_idx = np.argmin(simplex_f)
            x0 = simplex[best_idx].copy()
            f0 = simplex_f[best_idx]

            # Check convergence
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break

        return x0, f0
# EVOLVE-BLOCK-END
