"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive covariance ES + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start CMA-ES-like adaptive search
        # Use a population-based approach with adaptive step sizes
        n_starts = 5
        per_start_budget = max(1, (budget - 50) // n_starts)  # reserve 50 for NM

        for s in range(n_starts):
            if eval_count >= budget - 50:
                break
            if time.time() > deadline - 30:
                break

            # Initialize around a diverse point
            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Adaptive CMA-ES-like loop
            x = x0.copy()
            fx = eval_point(x)
            sigma = 1.5
            C = np.eye(dim)  # covariance
            # Track progress
            progress_window = []
            no_improve = 0

            for step in range(per_start_budget):
                if eval_count >= budget - 50:
                    break
                if time.time() > deadline - 30:
                    break

                # Generate candidate with adaptive covariance
                # Cholesky decomposition for correlated steps
                try:
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                z = rng.standard_normal(dim)
                y = x + sigma * L @ z
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)

                if fy < fx:
                    # Success: update mean, adapt covariance
                    direction = (y - x) / (sigma * np.linalg.norm(L @ z) + 1e-12)
                    C = 0.9 * C + 0.1 * np.outer(direction, direction)
                    # Normalize C to maintain unit trace
                    C /= np.trace(C) / dim
                    fx = fy
                    x = y.copy()
                    sigma *= 1.2
                    no_improve = 0
                    progress_window.append(fy)
                else:
                    sigma *= 0.85
                    no_improve += 1
                    progress_window.append(fx)

                # If stuck, try a larger random perturbation
                if no_improve > 15 and step % 5 == 0:
                    y2 = x + rng.uniform(-2, 2, dim)
                    y2 = np.clip(y2, -5.0, 5.0)
                    fy2 = eval_point(y2)
                    if fy2 < fx:
                        fx = fy2
                        x = y2.copy()
                        sigma = 1.0
                        no_improve = 0

        # Phase 2: Nelder-Mead refinement from best point
        if best_x is not None and eval_count < budget - 5:
            if time.time() < deadline - 10:
                nm_x, nm_f = self._nelder_mead(func, best_x.copy(), eval_count, budget, deadline, dim)
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x

        return best_x, best_f

    def _nelder_mead(self, func, x0, eval_count, budget, deadline, dim):
        """Nelder-Mead simplex method for final refinement."""
        t_start = time.time()
        # Initialize simplex
        alpha = 0.5  # step for new points
        simplex = [x0.copy()]
        fvals = [func(x0)]
        eval_count += 1

        for i in range(1, dim + 1):
            xi = x0.copy()
            xi[i - 1] += alpha if xi[i - 1] < 4.5 else -alpha
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            fvals.append(func(xi))
            eval_count += 1
            if eval_count >= budget or time.time() > deadline - 5:
                break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Nelder-Mead iterations
        max_iter = min(budget - eval_count, 200)
        for _ in range(max_iter):
            if eval_count >= budget or time.time() > deadline - 5:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            # Reflection
            xr = centroid + 1.0 * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            eval_count += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                eval_count += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = func(xc)
                    eval_count += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func(simplex[i])
                            eval_count += 1
                            if eval_count >= budget or time.time() > deadline - 5:
                                break
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = func(simplex[i])
                        eval_count += 1
                        if eval_count >= budget or time.time() > deadline - 5:
                            break

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Convergence check
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

        return simplex[0], fvals[0]
# EVOLVE-BLOCK-END
