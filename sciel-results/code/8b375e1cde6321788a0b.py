"""L-SMA optimizer with restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """L-SMA (success-weighted adaptation) with restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(123)

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = func(z)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # Phase 1: Latin hypercube initial exploration
        n_explore = max(10, budget // 5)
        if calls < budget:
            # Latin hypercube: divide each dimension into n_explore bins, pick one point per bin
            pts = np.zeros((n_explore, dim))
            for d in range(dim):
                bins = np.linspace(-5.0, 5.0, n_explore + 1)
                # One random point in each bin
                for i in range(n_explore):
                    pts[i, d] = bins[i] + (bins[i + 1] - bins[i]) * rng.random()
            # Add the center
            pts = np.vstack([pts, np.zeros(dim)])
            for pt in pts:
                if calls >= budget:
                    break
                evaluate(pt)

        # Phase 2: L-SMA with restarts
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = max(2, lambda_ // 2)
        # Weights: linear ranking
        w = np.arange(mu, 0, -1, dtype=float)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # L-SMA adaptation rate
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_1 = 2 / (dim + 1.3) ** 2
        c_mu = 2 / (dim + 1.3) ** 2 * min(1, (mu_eff - 2 + 1) / (dim + 2))

        def lsma_iteration(x, sigma, C, n_iter):
            """Run L-SMA for n_iter generations, returning updated state."""
            nonlocal best_x, best_f, calls
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for _ in range(n_iter):
                if calls >= budget or time.time() > deadline - 15.0:
                    break

                # Generate population
                pop = []
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline - 15.0:
                        break
                    z = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
                    f = evaluate(z)
                    pop.append((z, f))
                if len(pop) == 0:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[1])
                selected = pop[:mu]
                yk = np.array([s[0] for s in selected])

                # Weighted mean
                x_new = np.sum(w[:, None] * yk, axis=0)

                # Evolution path for sigma
                y_sigma = np.sum(w[:, None] * (yk - x) / (sigma + 1e-30), axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 1)) / chi_n < (1.4 + 2 / (dim + 1))
                sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Evolution path for C
                p_c = (1 - (1 - c_1) ** 1) * p_c + hsig * np.sqrt(c_1 * (2 - c_1) * mu_eff) * (x_new - x) / sigma
                x = x_new

                # L-SMA covariance update: success-weighted outer products
                delta_h = (1 - hsig) * c_1 * (2 - c_1)
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
                for i in range(mu):
                    diff = (yk[i] - x) / sigma
                    C += c_mu * w[i] * np.outer(diff, diff)

                C = 0.5 * (C + C.T)
                sigma = float(np.clip(sigma, 1e-12, 50.0))

            return x, sigma, C

        # Main L-SMA loop with restarts
        x = best_x.copy() if best_x is not None else np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        best_f_prev = best_f
        stagnation_counter = 0
        max_stagnation = 20

        while calls < budget and time.time() < deadline - 20.0:
            # Run a batch of L-SMA iterations
            batch_size = max(1, min(10, (budget - calls) // lambda_))
            x, sigma, C = lsma_iteration(x, sigma, C, batch_size)

            # Check for stagnation
            if best_f < best_f_prev - 1e-10:
                stagnation_counter = 0
                best_f_prev = best_f
            else:
                stagnation_counter += 1

            # Restart if stagnant
            if stagnation_counter >= max_stagnation and calls < budget - 30:
                x = best_x.copy()
                sigma = max(0.5, sigma * 0.5)
                C = np.eye(dim)
                stagnation_counter = 0
                best_f_prev = best_f
                # Small exploration burst
                for _ in range(min(5, budget - calls)):
                    z = x + 0.5 * rng.standard_normal(dim)
                    evaluate(z)

        # Phase 3: Local refinement
        if calls < budget - 5 and time.time() < deadline - 3.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for _ in range(min(40, budget - calls)):
                if calls >= budget or time.time() > deadline - 1.0:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline - 1.0:
                        break
                    # Positive
                    x_try = x.copy()
                    x_try[d] += step
                    x_try = clip(x_try)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fx:
                        x = x_try
                        fx = f_try
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Negative
                    if calls < budget:
                        x_try = x.copy()
                        x_try[d] -= 2 * step
                        x_try = clip(x_try)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < fx:
                            x = x_try
                            fx = f_try
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                if improved:
                    step *= 1.05
                else:
                    step *= 0.6
                if step < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
