"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 300s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        cc = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        cs = (lambda_ + 2 + 2 / (n + lambda_)) / (n + lambda_ + 5 + 2 * lambda_ / n)
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c1 = (2 + 2 / (mu_eff - 1) + mu_eff / (n + 2)) / ((n + 2) + 2 / (mu_eff - 1) + mu_eff)
        cmu = min(1 - c1,
                  (2 * (mu_eff - 2 + 1 / mu_eff) / (n + 1) ** 2) *
                  (c1 + 2 / (mu_eff - 1)))

        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with sigma0, using at most max_evals function evaluations."""
            x = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            c_sigma = 1.0 / np.sqrt(n + 2)
            c_sigma_sqrt = np.sqrt((1 + c_sigma) * (1 - c_sigma) * (1 + 2 * (n - 2) / (n + 2) ** 2))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            best_x = x.copy()
            best_f = np.inf
            evals = 0

            for gen in range(max_evals // lambda_ + 1):
                if evals >= max_evals:
                    break
                if time.time() > deadline:
                    break

                # Generate lambda samples
                B = np.linalg.cholesky(C)
                ys = []
                xs = []
                for i in range(lambda_):
                    y = rng.standard_normal(n)
                    x_i = np.clip(x + sigma * (B @ y), -5, 5)
                    ys.append(y)
                    xs.append(x_i)

                # Evaluate
                fs = []
                for x_i in xs:
                    if evals >= max_evals or time.time() > deadline:
                        break
                    f_i = func(x_i)
                    fs.append(f_i)
                    evals += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()

                if len(fs) < lambda_:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys = np.array(ys)
                xs = np.array(xs)
                fs = np.array(fs)
                ys = ys[order]
                xs = xs[order]
                fs = fs[order]

                # Mean of best mu
                m = np.sum(weights[:, None] * xs[:mu], axis=0)

                # Update step size control variable
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (ys[:mu] @ weights) / sigma)
                hsig = np.sqrt((1 - (1 - cs) ** (2 * (gen + 1))) / (1 - (1 - cs) ** 2)) * chi_n
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * hsig:
                    hc = 1.0
                else:
                    hc = 0.0

                # Update evolution path
                pc = (1 - cc) * pc + hc * np.sqrt(cc * (2 - cc) * mu_eff) * (m - x) / sigma
                x = m

                # Update covariance matrix
                delta = (1 + 2 * max(0, (hc - 1) * cc / c1) - c1)
                outer = np.outer(pc, pc)
                rank1 = c1 * (delta * C - outer)
                rankmu = cmu * np.sum(weights[i] * np.outer(ys[i] - pc / sigma, ys[i] - pc / sigma)
                                       for i in range(mu))
                C = C + rank1 + rankmu

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += (1e-12 - eigvals.min()) * np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 1e10))

                # Check convergence
                if np.linalg.norm(pc) / np.sqrt(np.trace(C)) / sigma < 1e-8:
                    break

            return best_x, best_f, evals

        # Multi-restart strategy
        best_x = None
        best_f = np.inf
        total_evals = 0

        # Initial diverse starting points
        start_points = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-5, 5, n),
            np.full(n, 4.20968746),  # Schwefel global optimum coordinate
            -np.full(n, 4.20968746),
            rng.uniform(-5, 5, n),
            np.zeros(n) + rng.uniform(-1, 1, n),
        ]

        # Evaluate each starting point once
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            f_sp = func(sp)
            total_evals += 1
            if f_sp < best_f:
                best_f = f_sp
                best_x = sp.copy()
            if total_evals >= budget or time.time() > deadline:
                break

        # Allocate remaining budget among restarts
        remaining = budget - total_evals
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # Sort start points by their initial fitness
        start_fits = []
        for sp in start_points[:total_evals]:
            pass  # We already evaluated them

        # Do CMA-ES from the best starting point first
        num_restarts = min(len(start_points), max(1, int(np.sqrt(remaining / 50))))
        evals_per_restart = max(20, remaining // num_restarts)

        for i in range(num_restarts):
            if total_evals >= budget or time.time() > deadline:
                break

            sp = start_points[i]
            sp = np.clip(sp, -5, 5)
            # Use a reasonable sigma based on problem scale
            sigma0 = 1.5 if i == 0 else 2.0

            max_evals_this = min(evals_per_restart, budget - total_evals)
            x_best, f_best, used = cma_es(sp, sigma0, max_evals_this)
            total_evals += used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

            # If we've found a very good solution, do local refinement
            if best_f < 100 and total_evals < budget and time.time() < deadline:
                # Local search around best point
                local_sigma = 0.5
                for _ in range(min(30, budget - total_evals)):
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + local_sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    total_evals += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        local_sigma *= 1.2
                    else:
                        local_sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
