"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Full CMA-ES with population-wide stagnation-triggered diverse restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_params(self, dim):
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
        return dict(lambda_=lambda_, mu=mu, w=w, mu_eff=mu_eff, c_sigma=c_sigma,
                    c_c=c_c, c_1=c_1, c_mu=c_mu, d_sigma=d_sigma, chi_n=chi_n)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        P = self._cma_params(dim)
        lambda_ = P["lambda_"]
        mu = P["mu"]
        w = P["w"]
        mu_eff = P["mu_eff"]
        c_sigma, c_c, c_1, c_mu, d_sigma, chi_n = (
            P["c_sigma"], P["c_c"], P["c_1"], P["c_mu"], P["d_sigma"], P["chi_n"])

        best_x = None
        best_fx = np.inf
        calls = 0

        def eval_point(y):
            nonlocal best_x, best_fx, calls
            fy = func(y)
            calls += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            return fy

        # Initial diverse sampling: capture multiple basins early for AOCC
        for _ in range(min(10, self.budget)):
            y = rng.uniform(-5, 5, dim)
            eval_point(y)
        if calls >= self.budget:
            return best_x, best_fx

        # Stagnation detection
        window = 3 * lambda_
        best_at_window_start = best_fx
        calls_at_window_start = calls
        restarts = 0

        while calls < self.budget and time.time() < deadline:
            # Initialize / re-initialize CMA-ES state
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
                eval_point(x)
            else:
                x = best_x.copy()
            sigma = 2.0
            C = np.eye(dim)
            D = np.ones(dim)
            B = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Run CMA-ES until stagnation or budget/time
            while calls < self.budget and time.time() < deadline:
                remaining = self.budget - calls
                lam_cur = min(lambda_, remaining)
                if lam_cur < 2:
                    break

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

                if len(ys) < mu:
                    break

                idx = np.argsort(fs)[:mu]
                ys_sorted = [ys[i] for i in idx]

                x_old = x.copy()
                x = sum(w[i] * ys_sorted[i] for i in range(mu))

                ps = (1 - c_sigma) * ps + \
                    np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((x - x_old) / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** ((calls + 1) // lambda_))) < \
                    (1.4 + 2 / (dim + 1)) * chi_n

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                pc = (1 - c_c) * pc + hsig * \
                    np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + \
                    c_mu / sigma ** 2 * sum(w[i] * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old)
                                            for i in range(mu))

                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-12))
                    D = D2
                    B = B2
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)

            # Stagnation check: population best not improving over the window
            improved = best_at_window_start - best_fx
            if improved < 1e-6 and calls - calls_at_window_start >= window and restarts < 4:
                # Diverse restart: fresh mean near the global best, identity covariance, larger sigma
                x = best_x.copy() + rng.normal(0, 1.5, dim)
                x = np.clip(x, -5, 5)
                restarts += 1
            else:
                # Reset the window regardless
                best_at_window_start = best_fx
                calls_at_window_start = calls

        return best_x, best_fx
# EVOLVE-BLOCK-END
