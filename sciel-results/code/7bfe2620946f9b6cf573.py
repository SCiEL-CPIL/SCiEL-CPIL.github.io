import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- CMA-ES implementation (simplified, robust for ridges) ----
        def cma_es(x0, sigma0, max_iter):
            nonlocal calls, best_f, best_x, deadline, budget
            rng = np.random.default_rng(123)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1.0/(4*dim) + 1.0/(20*dim*dim))
            cs = (2 + 2.0/min(1.0, 1.0/dim)) / (dim + 4)
            ds = 1 + 2*max(0.0, np.sqrt((dim+1)/3 - 1)) + cs
            cmu = 2 / ((dim+1.3)**2 + 10)
            c1 = (4 + dim/8) / ((dim+2)**2 + 4)
            cmu_ = cmu * min(1, (2 - c1) / (1 + c1))
            c_sigma = (mu_eff + 5) / (dim + mu_eff + 5)
            d_sigma = 1 + 2*max(0.0, np.sqrt((dim-1)/3 - 1)) + c_sigma
            mu = dim // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                # Generate lambda = 2*mu candidates
                ys = []
                zs = []
                for i in range(mu):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    ys.append(y)
                    zs.append(z)

                # Evaluate
                xs = [x + sigma * y for y in ys]
                fs = [f(xi) for xi in xs]

                # Sort
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                zs = [zs[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()

                # Update mean
                x_new = x + sigma * np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)

                # Update C
                delta_c = (x_new - x) / sigma
                C = (1 - c1 - cmu_) * C + c1 * (np.outer(delta_c, delta_c) + (1 - hsig * c_sigma * (1 - c_sigma)) * np.eye(dim))
                # hsig approximation
                hsig = 1.0 if np.linalg.norm(np.sum([weights[i] * zs[i] for i in range(mu)], axis=0)) / np.sqrt(1 - (1 - np.sum(weights))**2) / chi_n < (1.4 + 2/(dim+1)) else 0.0

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                B = eigvecs
                D = np.sqrt(eigvals)

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + hsig * c_sigma * np.sum([weights[i] * zs[i] for i in range(mu)], axis=0)
                sig_norm = np.linalg.norm(p_sigma)
                hsig = 1.0 if sig_norm / np.sqrt(1 - (1 - c_sigma)**(2*(it+1))) / chi_n < (1.4 + 2/(dim+1)) else 0.0

                # Update p_c
                p_c = (1 - cs) * p_c + hsig * cs * delta_c
                hc = 1.0 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs)**(2*(it+1))) / chi_n) < (1.4 + 2/(dim+1)) else 0.0
                C = (1 + (2*(hc - 1) * c1) * np.outer(p_c, p_c)) * C

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (sig_norm / chi_n - 1))

                x = x_new

            return x, best_f

        # ---- Multi-start CMA-ES ----
        rng0 = np.random.default_rng(7)

        # Quick random scan to find a good starting region
        candidates = []
        n_starts = min(5, budget // 100)
        for i in range(n_starts):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng0.uniform(-4.0, 4.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try))

        candidates.sort(key=lambda t: t[0])

        # Run CMA-ES from the best starting point
        if candidates:
            x_start = candidates[0][1].copy()
            sigma0 = 2.0
            # Run CMA-ES with enough iterations to use remaining budget
            max_iter = min(200, (budget - calls) // mu if mu > 0 else 100)
            x_final, fx_final = cma_es(x_start, sigma0, max_iter)

        return best_x, best_f
