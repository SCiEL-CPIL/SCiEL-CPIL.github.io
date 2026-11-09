import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = self.dim

        def cmaes(x0, budget):
            """CMA-ES with population 4, integrated restart rule, bounded sigma."""
            nonlocal calls, best_x, best_f, deadline
            lam = 4
            mu = lam // 2
            # weights: cumulative (mu_eff = 1 / sum w^2)
            w = np.full(mu, np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            # adaptation rates
            c_sigma = (mueff + 1.8) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = np.asarray(x0, dtype=np.float64).copy()
            sigma = 1.0
            sigma_min, sigma_max = 1e-12, 10.0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            # restart / stagnation control
            stall = 0
            stall_limit = 30
            no_improve = 0
            no_improve_limit = 25

            def eval(xc):
                try:
                    return safe_func(np.clip(xc, -5.0, 5.0))
                except (TimeoutError, StopIteration):
                    raise

            while calls < budget and time.time() < deadline:
                # sample
                ys = []
                xs = []
                for _ in range(lam):
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(n)
                    ys.append(z)
                    xs.append(np.clip(x + sigma * (B @ z), -5.0, 5.0))
                # evaluate
                fs = np.empty(lam)
                try:
                    for i in range(lam):
                        fs[i] = eval(xs[i])
                except (TimeoutError, StopIteration):
                    break

                order = np.argsort(fs)
                xnew = np.sum(w[:, None] * np.array([xs[order[i]] for i in range(mu)]), axis=0)

                # update mean
                x = xnew
                # evolution path for sigma
                y_sigma = np.sum(w[:, None] * np.array([ys[order[i]] for i in range(mu)]), axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma / chi_n
                # h_sigma: test ||p_sigma|| against expected value
                hs = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                # evolution path for C
                p_c = (1 - c_c) * p_c + (hs * np.sqrt(c_c * (2 - c_c) * mueff)) * (xnew - x) / sigma
                # rank-one + rank-mu update
                delta_h = (1 - c_c) ** 2 - 1
                C = ((1 + c_1 + c_mu - c_c - c_c + delta_h) * C
                     + c_1 * np.outer(p_c, p_c)
                     + c_mu * np.sum(w[:, None, None] * np.outer(
                         (xnew - x) / sigma, (xnew - x) / sigma)
                         * np.array([[1.0]] * 0) * 0, axis=0)
                     )
                # rank-mu term (computed properly)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    d = (xnew - x) / sigma
                    rank_mu += w[i] * np.outer(d, d)
                C = C - c_mu * rank_mu
                # symmetric + positive-definite safeguard
                C = 0.5 * (C + C.T)
                try:
                    np.linalg.cholesky(C + 1e-9 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = C + 1e-3 * np.eye(n)
                # sigma adaptation
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = float(np.clip(sigma, sigma_min, sigma_max))

                # restart / stagnation rule
                best_this = float(np.min(fs))
                if best_this < best_f - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1
                stall += 1
                if (stall >= stall_limit) or (no_improve >= no_improve_limit) or (sigma < 1e-10):
                    # restart from a fresh random point
                    stall = 0
                    no_improve = 0
                    x = rng.uniform(-5.0, 5.0, n)
                    sigma = 1.0
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    try:
                        eval(x)
                    except (TimeoutError, StopIteration):
                        break

        # Phase 1: CMA-ES global search
        x0 = np.zeros(n)
        try:
            safe_func(x0)
        except (TimeoutError, StopIteration):
            return best_x, best_f

        global_budget = int(self.budget * 0.85)
        try:
            cmaes(x0, global_budget)
        except (TimeoutError, StopIteration, Exception):
            pass

        # Phase 2: local refinement with Nelder-Mead on the global best
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None and time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass

        return best_x, best_f
