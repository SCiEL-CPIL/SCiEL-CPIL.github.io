"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with population 4, restart rule, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_time():
            return time.time() >= deadline

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial point
        x0 = rng.uniform(-4, 4, dim)
        fx0 = evaluate(x0)

        # CMA-ES parameters
        lam = 4  # population size
        mu = lam // 2  # number of weights
        # Weights: linearly decreasing, normalized
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w = w[:lam]  # only first mu have non-zero weight
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial sigma
        sigma = 2.0
        x_mean = x0.copy()
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        alpha_cs = 2 * (c_sigma + 0.5) * c_sigma
        # Precompute C^1/2 via Cholesky
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)

        # Restart threshold
        restart_count = 0
        max_restarts = 3
        stagnation = 0

        for gen in range(budget - calls_used):
            if check_time():
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                if check_time():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                ys.append(y)
                fs.append(fy)

            if len(ys) < lam:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update mean
            x_mean_new = np.sum(w[:, None] * ys[:mu], axis=0)

            # Update covariance
            y_diffs = (ys[:mu] - x_mean) / sigma
            delta_s = 0.0
            for i in range(mu):
                delta_s += w[i] * (np.dot(y_diffs[i], y_diffs[i]) - dim)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_diffs @ w)
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            ps_norm = np.linalg.norm(ps)
            if ps_norm > (1.4 + 1 / c_sigma) * chi_n:
                pc = (1 - c_c) * pc
            else:
                pc = (1 - c_c) * pc + c_c * (y_diffs @ w)

            hsig = (1 - np.prod(1 - c_sigma) ** (1 / (dim + 5))) * \
                   (np.dot(pc, pc) / (1 - np.prod(1 - c_c) ** 2) - 1) / 2
            if hsig < 1:
                C = (1 - c_1 - c_mu) * C + \
                    c_1 * (np.outer(pc, pc) - C + 2 * c_c * (1 - c_c) * C) + \
                    c_mu * np.sum(w[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)

            # Symmetrize
            C = 0.5 * (C + C.T)
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal
                C += 1e-8 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)

            x_mean = x_mean_new
            x_mean = np.clip(x_mean, -5, 5)

            # Restart check: if no improvement for many generations
            if fs[0] >= best_f:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation > 30 and restart_count < max_restarts:
                restart_count += 1
                stagnation = 0
                sigma = 2.0
                x_mean = best_x.copy() if best_x is not None else rng.uniform(-4, 4, dim)
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        # Nelder-Mead polish on best point
        if best_x is not None and not check_time():
            nm_x = best_x.copy()
            nm_f = best_f
            # Initial simplex
            simplex = [nm_x.copy()]
            for i in range(dim):
                xi = nm_x.copy()
                step = 0.5 if abs(nm_x[i]) < 1 else 0.1
                if nm_x[i] + step > 5:
                    step = -step
                xi[i] += step
                simplex.append(xi)

            for i in range(dim):
                simplex[i+1] = evaluate(simplex[i+1]) if False else simplex[i+1]
                # Evaluate simplex points
            # Re-evaluate all simplex points
            simplex_f = [evaluate(simplex[0])]
            for i in range(1, dim + 1):
                simplex_f.append(evaluate(simplex[i]))

            # Nelder-Mead iterations
            for it in range(200):
                if check_time():
                    break
                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = evaluate(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[0] = evaluate(simplex[0])
                        for i in range(1, dim + 1):
                            simplex_f[i] = evaluate(simplex[i])

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
