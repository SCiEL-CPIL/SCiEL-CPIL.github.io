import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts + Nelder-Mead polish for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        def check_time():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # CMA-ES parameters
        lambda_ = 4
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        d_sigma = 1 + 0.3 * min((mu_eff - 1) / 2, dim / 2)
        d_cov = 2 + 2 * min(mu_eff / (dim + 2), 1) - 1 / (dim + 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_cov = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov_neg = (2 - c_cov + c_sigma) / (dim + 2)

        # Phase 1: CMA-ES with restarts
        max_evals_per_run = 120
        restart_count = 0
        max_restarts = 8

        while total_calls < budget and restart_count <= max_restarts and check_time():
            # Initialize CMA-ES state
            if restart_count == 0:
                m = rng.uniform(-5, 5, dim)
            else:
                m = best_x.copy() + rng.normal(0, 0.5, dim)
                m = np.clip(m, -5, 5)

            sigma = 2.0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Evaluate initial mean
            if total_calls < budget and check_time():
                fx = func(m)
                total_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = m.copy()

            run_calls = 0
            no_improve = 0
            prev_best = best_f

            while run_calls + lambda_ < budget and total_calls + lambda_ < budget and check_time():
                # Generate population
                pop = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    x = m + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    pop.append(x)

                # Evaluate population
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    fvals[i] = func(pop[i])
                    total_calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                run_calls += lambda_

                # Sort by fitness
                idx = np.argsort(fvals)
                y_sorted = np.array([pop[i] for i in idx])

                # Update mean
                m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

                # Update sigma
                y_mean = C_inv_sqrt @ (m_new - m)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                norm_ps = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 50)

                # Update covariance
                h_sigma = norm_ps < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                p_c = (1 - c_cov) * p_c + h_sigma * np.sqrt(c_cov * (2 - c_cov) * mu_eff) * y_mean

                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (y_sorted[i] - m)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_cov * c_cov_neg - c_cov) * C + c_cov * np.outer(p_c, p_c) + c_cov_neg * rank_mu

                # Ensure C is positive definite
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_sqrt = eigvecs @ np.sqrt(np.diag(eigvals))
                    C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

                m = m_new

                # Track improvement
                if best_f < prev_best - 1e-10:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1

                # Restart if stuck
                if no_improve > 25:
                    break

            restart_count += 1

        # Phase 2: Nelder-Mead polish
        if check_time() and total_calls < budget - 10:
            # Build initial simplex
            simplex = [best_x.copy()]
            fvals_simplex = [best_f]
            for i in range(dim):
                xi = best_x.copy()
                step = 0.5 if best_x[i] > -4.5 else 0.5
                xi[i] += step
                xi[i] = np.clip(xi[i], -5, 5)
                fi = func(xi)
                total_calls += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                simplex.append(xi.copy())
                fvals_simplex.append(fi)

            # Nelder-Mead iterations
            max_nm_iters = 50
            for _ in range(max_nm_iters):
                if total_calls >= budget or not check_time():
                    break

                # Sort simplex
                order = np.argsort(fvals_simplex)
                simplex = [simplex[i] for i in order]
                fvals_simplex = [fvals_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                total_calls += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    total_calls += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals_simplex[-1] = f_ref
                elif f_ref < fvals_simplex[-2]:
                    simplex[-1] = x_ref
                    fvals_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    total_calls += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        fvals_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals_simplex[i] = func(simplex[i])
                            total_calls += 1
                            if fvals_simplex[i] < best_f:
                                best_f = fvals_simplex[i]
                                best_x = simplex[i].copy()
                        if total_calls >= budget or not check_time():
                            break

        return best_x, best_f
