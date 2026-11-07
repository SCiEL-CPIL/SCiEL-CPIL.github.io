"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES + Hessian Newton polishing for BBOB f11 (Discus).
    CMA-ES adaptively explores the anisotropic landscape, then
    Newton polishing exploits the local quadratic structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit
        calls_made = 0
        x_best = np.zeros(dim)
        f_best = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def update_best(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                x_best = x.copy()
                f_best = f

        # Evaluate origin
        f0 = func(np.zeros(dim))
        calls_made += 1
        update_best(np.zeros(dim), f0)

        # --- Phase 1: CMA-ES ---
        # CMA-ES parameters
        lambda_ = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        sigma = 2.0  # initial step size
        x_mean = np.zeros(dim)
        C = np.eye(dim)  # covariance matrix
        C_sqrt = np.eye(dim)  # Cholesky-like (we use eigendecomposition)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Strategy parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # History of best values for restart detection
        best_history = [f0]
        no_improve_count = 0
        restart_threshold = 30  # generations without improvement before restart

        for gen in range(200):  # max generations
            if time.time() > deadline or calls_made >= budget - 80:
                break

            # Generate lambda samples
            # Update C_sqrt via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Sample
            ys = []
            xs = []
            for i in range(lambda_):
                z = np.random.randn(dim)
                y = C_sqrt @ z
                x = x_mean + sigma * y
                x = clip(x)
                ys.append(y)
                xs.append(x)

            # Evaluate
            fs = []
            for x in xs:
                if calls_made >= budget:
                    break
                f_val = func(x)
                calls_made += 1
                fs.append(f_val)
                update_best(x, f_val)

            if len(fs) < lambda_:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt.T @ (y_w / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            p_c = (1 - c_c) * p_c + c_c * (np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma) * y_w

            # Update covariance
            delta_hsig = (1 + min(0, (norm_p_sigma / chi_n - 1) * d_sigma)) * c_sigma
            rank_one = np.outer(p_sigma, p_sigma)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])

            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * (rank_one + (delta_hsig - 1) * C) + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            x_mean = clip(x_mean + sigma * y_w)

            # Check for stagnation
            if fs[0] < best_history[-1] * 0.999 or fs[0] < best_history[-1] - 1e-6:
                no_improve_count = 0
            else:
                no_improve_count += 1

            best_history.append(fs[0])
            if len(best_history) > 50:
                best_history.pop(0)

            # Restart if stuck
            if no_improve_count >= restart_threshold:
                # Restart with smaller sigma around current best
                sigma = max(sigma * 0.5, 0.1)
                x_mean = x_best.copy()
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                best_history = [f_best]

            # If very close to optimum, break early
            if fs[0] - f_best < 1e-10:
                break

        # --- Phase 2: Hessian Newton Polishing ---
        x_curr = x_best.copy()
        f_curr = f_best
        h = 0.05

        while calls_made < budget - 5 and time.time() < deadline:
            # 1. Estimate Gradient
            grad = np.zeros(dim)
            for i in range(dim):
                if calls_made + 2 > budget:
                    break
                xp = x_curr.copy(); xp[i] += h
                xm = x_curr.copy(); xm[i] -= h
                xp = clip(xp); xm = clip(xm)
                fpx = func(xp); fxm = func(xm)
                calls_made += 2
                grad[i] = (fpx - fxm) / (2 * h)

            if np.linalg.norm(grad) < 1e-10:
                break

            # 2. Estimate Hessian
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    if calls_made + 4 > budget:
                        break
                    if i == j:
                        xp = x_curr.copy(); xp[i] += h
                        xm = x_curr.copy(); xm[i] -= h
                        xp = clip(xp); xm = clip(xm)
                        fpp = func(xp); fxm = func(xm)
                        calls_made += 2
                        H[i, i] = (fpp - 2 * f_curr + fxm) / (h * h)
                    else:
                        xpp = x_curr.copy(); xpp[i] += h; xpp[j] += h
                        xpm = x_curr.copy(); xpm[i] += h; xpm[j] -= h
                        xmp = x_curr.copy(); xmp[i] -= h; xmp[j] += h
                        xmm = x_curr.copy(); xmm[i] -= h; xmm[j] -= h
                        xpp = clip(xpp); xpm = clip(xpm)
                        xmp = clip(xmp); xmm = clip(xmm)
                        fpp = func(xpp); fpm = func(xpm)
                        fmp = func(xmp); fxm = func(xmm)
                        calls_made += 4
                        H[i, j] = (fpp - fpm - fmp + fxm) / (4 * h * h)
                        H[j, i] = H[i, j]

            # 3. Compute Newton Step with regularization
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals_reg = np.maximum(eigvals, 1e-6)
            H_reg = (eigvecs @ np.diag(eigvals_reg) @ eigvecs.T)

            try:
                step = np.linalg.solve(H_reg, grad)
            except np.linalg.LinAlgError:
                step = -grad * 0.1

            # 4. Line search
            alpha = 1.0
            improved = False
            for _ in range(15):
                if calls_made >= budget:
                    break
                x_trial = clip(x_curr - alpha * step)
                f_trial = func(x_trial)
                calls_made += 1

                if f_trial < f_curr - 1e-12:
                    x_curr = x_trial
                    f_curr = f_trial
                    update_best(x_curr, f_curr)
                    improved = True
                    break
                else:
                    alpha *= 0.5

            if not improved:
                h *= 0.5
                if h < 1e-5:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
