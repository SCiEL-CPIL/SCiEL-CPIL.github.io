import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.call_count = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        self.call_count = 0
        self.best_x = None
        self.best_f = float('inf')

        deadline = self.start_time + 320  # 10s safety margin before 330s kill

        def evaluate(x):
            """Evaluate a point, tracking calls and time."""
            if time.time() > deadline:
                raise TimeoutError("Time limit")
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            self.call_count += 1
            x = np.asarray(x, dtype=float)
            # Clip to box (func also clips, but be safe)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def cma_es(start, sigma, max_evals):
            """
            Run CMA-ES from a starting point.
            Returns (best_x, best_f) after max_evals or budget/time exhausted.
            """
            dim = self.dim
            # Population size: default is 4 + floor(3*ln(dim))
            lam = max(4, int(4 + 3 * np.log(dim)))
            # CMA-ES parameters
            c_sigma = (4 + lam / (2 * dim)) / ((dim + 1) **2 + 2)
            c_c = (4 + dim / 4) / ((dim + 1) **2 + 4)
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (c_c + 2)
            c_mu = (2 / ((dim + 1.3) ** 2 + mu_eff)) * np.clip(1 - c_1, 0, 0.5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # State
            mean = np.array(start, dtype=float)
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            best_x = mean.copy()
            best_f = float('inf')
            evals_used = 0

            while True:
                if time.time() > deadline:
                    break
                if self.call_count + lam > self.budget:
                    # Can't fit a full population, break
                    break

                # Generate population
                ys = []
                xs = []
                fs = []
                for i in range(lam):
                    if time.time() > deadline or self.call_count >= self.budget:
                        break
                    y = np.random.randn(dim)
                    x = mean + sigma * (C_sqrt @ y)
                    ys.append(y)
                    xs.append(x)
                    f = evaluate(x)
                    fs.append(f)
                    evals_used += 1

                if len(ys) == 0:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                # Update best
                for i in range(len(fs)):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xs[i].copy()

                # Use only first mu individuals
                mu_actual = min(mu, len(ys))
                if mu_actual < 2:
                    break

                # Update mean
                y_mean = np.zeros(dim)
                for i in range(mu_actual):
                    y_mean += w[i] * ys[i]
                mean = mean + c_sigma * sigma * y_mean

                # Update covariance
                y_w = np.array([ys[i] - y_mean for i in range(mu_actual)])
                delta_s = (sigma * np.linalg.norm(y_mean) / chi_n - 1) * c_sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta_s / sigma) * y_mean

                rank_mu = np.outer(y_w.T, w[:, None] * y_w.T)
                rank_one = np.outer(ps, ps)
                h_sigma = 1 if (np.sum(ps * ps) / (1 - (1 - c_sigma) ** (2 * (evals_used // lam + 1)))) < (1 + 2 / dim) * chi_n ** 2 else 0
                C = (1 + c_1 * (h_sigma - 1) + c_mu - c_1) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Regularize
                C += 1e-10 * np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update C_sqrt via eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except np.linalg.LinAlgError:
                    break

                # Convergence check
                if sigma < 1e-12 or np.all(np.abs(C) < 1e-12):
                    break

                # Stop if we've used enough evals
                if evals_used >= max_evals:
                    break

            return best_x, best_f

        # Phase 1: Quick random sampling near center for early AOCC
        n_initial = min(20, self.budget // 5)
        for i in range(n_initial):
            if time.time() > deadline or self.call_count >= self.budget:
                break
            x = np.random.uniform(-2.0, 2.0, size=self.dim)
            try:
                evaluate(x)
            except (TimeoutError, ValueError):
                break

        # Phase 2: CMA-ES from best point found
        if self.best_x is not None and self.call_count < self.budget and time.time() < deadline:
            remaining = self.budget - self.call_count
            # First CMA-ES: use ~60% of remaining budget
            max_evals_1 = max(50, int(remaining * 0.6))
            try:
                bx, bf = cma_es(self.best_x, 1.0, max_evals_1)
            except Exception:
                pass

        # Phase 3: If budget remains, do a second CMA-ES from a different point
        if self.call_count < self.budget and time.time() < deadline:
            remaining = self.budget - self.call_count
            if remaining > 40:
                # Try a different starting point
                start2 = np.random.uniform(-3.0, 3.0, size=self.dim)
                max_evals_2 = remaining
                try:
                    bx2, bf2 = cma_es(start2, 1.5, max_evals_2)
                except Exception:
                    pass

        return self.best_x, self.best_f
