"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with diverse initialization for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(y):
            nonlocal best_x, best_f, calls
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            return fy

        # Phase 1: Quick diverse sampling to find a good starting point
        # This helps AOCC by getting a reasonable value early
        for _ in range(min(10, self.budget)):
            y = rng.uniform(-5, 5, dim)
            eval_point(y)

        if calls >= self.budget or best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        D = np.ones(dim)
        B = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # CMA-ES parameters
        lambda_ = 4 * dim
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Stagnation tracking
        window = 3 * lambda_
        best_at_window_start = best_f
        calls_at_window_start = calls
        restarts = 0
        max_restarts = 5

        while calls < self.budget and time.time() < deadline:
            # Check for stagnation
            if calls - calls_at_window_start >= window:
                if best_f - best_at_window_start < 1e-6:
                    # Stagnation detected, try a restart
                    restarts += 1
                    if restarts > max_restarts:
                        break
                    # Reset CMA-ES state with a new random point
                    x = rng.uniform(-5, 5, dim)
                    eval_point(x)
                    sigma = 2.0
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    best_at_window_start = best_f
                    calls_at_window_start = calls
                    continue
                else:
                    # Progress made, reset window
                    best_at_window_start = best_f
                    calls_at_window_start = calls

            # Determine how many evaluations we can do in this generation
            remaining = self.budget - calls
            lam_cur = min(lambda_, remaining)
            if lam_cur < 2:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lam_cur):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                ys.append(y)
                fs.append(fy)

                if time.time() > deadline:
                    break

            # Sort by fitness
            if len(ys) < mu:
                break

            idx = np.argsort(fs)[:mu]
            ys_sorted = [ys[i] for i in idx]

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * ys_sorted[i] for i in range(mu))

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((x - x_old) / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** ((calls + 1) // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu / sigma ** 2 * sum(
                w[i] * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old) for i in range(mu))

            # Eigendecomposition
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                D = D2
                B = B2
            except np.linalg.LinAlgError:
                # Reset to identity if eigen decomposition fails
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
