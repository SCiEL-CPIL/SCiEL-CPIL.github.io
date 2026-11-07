"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive lambda and Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def nelder_mead(x0, fx0, max_iter=60):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if x0[i] < 4.5 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    return
                simplex.append(xi)
                fvals.append(fi)

            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(max_iter):
                if time.time() > deadline - 15.0 or calls_made >= self.budget:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                x_centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]

                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)
                if f_ref is None:
                    break

                if f_ref < fvals[0]:
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_call(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)
                    if f_con is None:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_call(simplex[i])
                            if fvals[i] is None:
                                return

        def cma_es(x0, sigma0, max_evals=350):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            mean = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            # Precompute sqrt of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # CMA-ES parameters
            mu = max(1, dim // 2)  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            evals_used = 0

            # Initial lambda
            lam = max(4, 2 * mu)

            while evals_used < max_evals and calls_made < self.budget and time.time() < deadline - 20.0:
                # Sample candidates
                ys = rng.standard_normal((lam, dim))
                xs = np.array([mean + sigma * (C_sqrt @ y) for y in ys])
                xs = np.clip(xs, -5.0, 5.0)

                # Evaluate
                fs = []
                for x in xs:
                    if calls_made >= self.budget or time.time() > deadline - 20.0:
                        break
                    fx = safe_call(x)
                    if fx is None:
                        return
                    fs.append(fx)

                if len(fs) < mu:
                    break

                fs = np.array(fs)
                xs_evaluated = xs[:len(fs)]
                evals_used += len(fs)

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs_evaluated[order]
                fs_sorted = fs[order]

                # Update mean
                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)

                # Update sigma
                y_sigma = (C_inv_sqrt @ (new_mean - mean)) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < 1.4 + 2 / (dim + 1)
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (new_mean - mean)) / sigma
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None, None] * np.outer(xs_sorted[:mu] - mean, xs_sorted[:mu] - mean), axis=0) / sigma ** 2
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                mean = new_mean

                # Adaptive lambda
                if evals_used % 50 == 0:
                    # Increase lambda if progress is good
                    if len(fs_sorted) > 1 and fs_sorted[0] < fs_sorted[-1] * 0.9:
                        lam = min(lam + 2, 20)
                    else:
                        lam = max(lam - 2, 4)

        # Phase 1: Multi-start with diverse initial points
        starts = []
        starts.append(np.zeros(self.dim))
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = 3.0
            starts.append(pt)
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = -3.0
            starts.append(pt)
        # Add some random starts
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Evaluate starts
        start_fvals = []
        for x_s in starts:
            if time.time() > deadline - 40.0 or calls_made >= self.budget:
                break
            fx_s = safe_call(x_s)
            if fx_s is None:
                break
            start_fvals.append((fx_s, x_s))

        start_fvals.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES on the best starting point
        if start_fvals and calls_made < self.budget and time.time() < deadline - 30.0:
            fx_best, x_best = start_fvals[0]
            cma_es(x_best, 2.0, max_evals=300)

        # Phase 3: Nelder-Mead polish from the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 20.0:
            nelder_mead(best_x.copy(), best_f, max_iter=50)

        # Phase 4: Final CMA-ES with small sigma for fine-tuning
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 10.0:
            cma_es(best_x.copy(), 0.3, max_evals=50)

        # Phase 5: Final Nelder-Mead polish
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5.0:
            nelder_mead(best_x.copy(), best_f, max_iter=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
