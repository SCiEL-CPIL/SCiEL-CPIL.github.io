"""Portfolio optimizer: exploration + population-based CMA-ES refinement for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer:
    1. Brief exploratory phase (random + structured points) to find promising basins.
    2. Population-based CMA-ES local refinement (adaptive covariance, mean shift).
    3. Multi-restart: several independent CMA-ES runs from diverse starts, keep global best.
    4. Time-bounded with explicit deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # ---------- Phase 1: Exploration ----------
        # Structured + random points to cover the box and find good basins.
        candidates = [np.zeros(dim)]
        n_explore = min(40, max(10, budget // 12))
        for _ in range(n_explore):
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        for x in candidates:
            if check_deadline():
                break
            x = np.clip(x, -5, 5)
            safe_eval(x)

        if best_x is None:
            x = np.zeros(dim)
            safe_eval(x)
            return best_x, best_f

        # ---------- Phase 2: CMA-ES local refinement (population-based) ----------
        # CMA-ES with adaptive covariance for the rotated/shifted Rastrigin landscape.
        def cma_es(x_start, f_start, n_iter):
            nonlocal calls_made, best_x, best_f
            if check_deadline() or calls_made >= budget:
                return

            # Population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            # Initial step size: moderate to escape local minima
            sigma = 0.8
            # Covariance: diagonal, isotropic
            C = np.eye(dim)
            # Mean
            m = x_start.copy()
            fm = f_start

            # CMA-ES adaptation parameters
            cmu = 1.0 / (4 * dim + lam)
            c1 = (2 / ((dim + 1) ** 2 + 10))
            cs = (dim + 1.7) / (dim + 9 + 2 * lam)
            # Path variables
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cs
            dampc = (1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)) * c1
            # Eigenvalues for C
            eigvals = np.ones(dim)
            # Inverse square root of C
            Cinv_sqrt = np.eye(dim)

            # Cholesky of C
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                Csqrt = np.eye(dim)

            for it in range(n_iter):
                if check_deadline() or calls_made >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = m + sigma * (Csqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_eval(y)
                    offspring.append((y, fy))

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])

                # Selection
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights ** 2)

                # New mean
                m_new = np.zeros(dim)
                for i in range(mu):
                    m_new += weights[i] * offspring[i][0]
                m_new = np.clip(m_new, -5, 5)

                # Update covariance
                # Path covariance
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (Cinv_sqrt @ (offspring[i][0] - m))
                pc = (1 - c1) * pc + dampc * c1 * y_w
                c1pc1 = c1 * (2 - c1)

                # Rank-one update
                diff = m_new - m
                C = (1 - c1pc1 - c1 * cmu) * C + c1 * (pc @ pc.T + c1pc1 * np.eye(dim)) \
                    + c1 * cmu * np.sum(weights[i] * np.outer(
                        (Cinv_sqrt @ (offspring[i][0] - m)),
                        (Cinv_sqrt @ (offspring[i][0] - m))) for i in range(mu))

                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size (1/5 success rule)
                sigma_old = sigma
                # Success rate: fraction of offspring better than parent
                success = sum(1 for i in range(mu) if offspring[i][1] < fm) / mu
                if success > 0.2:
                    sigma *= 1.05
                elif success < 0.1:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-5, 5.0)

                # Update Cinv_sqrt
                try:
                    Csqrt = np.linalg.cholesky(C)
                    Cinv_sqrt = np.linalg.inv(Csqrt)
                except np.linalg.LinAlgError:
                    Csqrt = np.eye(dim)
                    Cinv_sqrt = np.eye(dim)

                m = m_new
                fm = min(fm, offspring[0][1])

                # Stagnation check: if sigma is very small and no improvement, break
                if sigma < 1e-4:
                    break

        # ---------- Multi-restart CMA-ES ----------
        # Allocate remaining budget across several restarts.
        remaining = budget - calls_made
        if remaining > 0 and not check_deadline():
            n_restarts = 6
            per_restart = max(10, remaining // n_restarts)

            for r in range(n_restarts):
                if check_deadline() or calls_made >= budget:
                    break

                # Choose start point
                if r == 0:
                    x0 = best_x.copy()
                    f0 = best_f
                elif r % 2 == 1:
                    # Perturb best with moderate noise
                    x0 = best_x + rng.uniform(-1.5, 1.5, dim)
                    x0 = np.clip(x0, -5, 5)
                    f0 = safe_eval(x0)
                else:
                    # Random restart
                    x0 = rng.uniform(-4.5, 4.5, dim)
                    f0 = safe_eval(x0)

                # Run CMA-ES from this start
                cma_es(x0, f0, per_restart)

        # ---------- Phase 3: Final polish around global best ----------
        if best_x is not None and calls_made < budget and not check_deadline():
            polish_budget = min(30, budget - calls_made)
            cma_es(best_x, best_f, polish_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
