"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and local polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4 * d, 8)
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initialize
        x_mean = np.zeros(d)
        sigma = 2.0
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        C_sqrt = np.eye(d)
        C_inv = np.eye(d)
        invsqrtC = np.eye(d)

        def update_matrices():
            nonlocal C, C_sqrt, C_inv, invsqrtC
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_matrices()

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            calls += 1
            try:
                f = float(func(x))
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        fx = safe_eval(x_mean)
        if fx is None:
            return x_mean, np.inf

        # Main CMA-ES loop
        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            gen += 1

            # Generate offspring
            Y = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate offspring
            F = []
            for y in Y:
                fy = safe_eval(y)
                if fy is None:
                    F.append(np.inf)
                else:
                    F.append(fy)

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = [Y[i] for i in idx]
            F_sorted = [F[i] for i in idx]

            # Update mean
            y_w = np.sum([w[i] * Y_sorted[i] for i in range(mu)], axis=0)
            x_mean_old = x_mean.copy()
            x_mean = y_w

            # Update sigma (fixed: use calls+1 for the running average)
            z_w = invsqrtC @ (y_w - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_w
            # Correct h_sigma: use gen as the number of updates so far
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) < (1.4 + 2 / (d + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (y_w - x_mean_old) / sigma

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_c, p_c)
            rankmu = np.sum([w[i] * np.outer(Y_sorted[i] - x_mean_old, Y_sorted[i] - x_mean_old) for i in range(mu)], axis=0)
            C = (1 + c_1 * delta_h + c_mu - c_1) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            update_matrices()

            # Ensure x_mean is in bounds
            x_mean = np.clip(x_mean, -5, 5)

            # Stop early if sigma is very small (converged)
            if sigma < 1e-8:
                break

        # Local polish: use remaining budget for coordinate descent / Nelder-Mead
        if best_x is not None and calls < budget and time.time() < deadline:
            x_best = best_x.copy()
            f_best = best_f
            remaining = budget - calls

            # Nelder-Mead style local search
            eps = 0.1
            alpha = 1.0
            rho = -0.5
            gamma = 2.0

            # Build simplex around best point
            simplex = [x_best.copy()]
            for i in range(d):
                pt = x_best.copy()
                pt[i] += eps
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            # Evaluate simplex
            simplex_f = []
            for pt in simplex:
                f = safe_eval(pt)
                if f is None:
                    simplex_f.append(f_best)
                else:
                    simplex_f.append(f)

            # Nelder-Mead iterations
            for _ in range(min(remaining // (d + 1), 50)):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5, 5)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_eval(x_ref)
                if f_ref is None:
                    break
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_eval(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_eval(x_con)
                    if f_con is None:
                        break
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_sh = safe_eval(simplex[i])
                            if f_sh is None:
                                break
                            simplex_f[i] = f_sh
                        if f_sh is None:
                            break

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < f_best:
                    f_best = simplex_f[min_idx]
                    x_best = simplex[min_idx].copy()

        return x_best if best_x is not None else x_mean, f_best if best_x is not None else best_f
# EVOLVE-BLOCK-END
