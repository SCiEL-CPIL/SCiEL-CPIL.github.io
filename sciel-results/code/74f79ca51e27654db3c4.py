"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with early-progress focus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            fx = float(fx)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # CMA-ES parameters
        # For dim=5, use lambda = 4 + floor(3 * log(dim)) = 4 + 7 = 11
        lam = 4 + int(np.floor(3 * np.log(dim)))
        mu = lam // 2

        # Strategy parameters
        alpha_mu = 2.0 / (mu + 1)
        alpha_sigma = 0.8 + 0.5 * (mu - 2) / (dim + mu - 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + alpha_sigma
        cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        cs = (mu + 2 + alpha_sigma / 2) / (dim + mu + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mu)) * (2 - alpha_mu / 2)
        cmu = c1 * (mu - 2 + alpha_mu / c1)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1)) + alpha_sigma

        # Initial state: start near origin with moderate sigma
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # Not strictly needed for small dim
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        def cma_es_step(x_mean, sigma, C, pc, ps):
            """One iteration of CMA-ES."""
            nonlocal calls_made, best_x, best_f, deadline

            # Generate offspring
            # Use Cholesky-like decomposition via eigendecomposition for small dim
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            Y = np.random.default_rng().standard_normal((lam, dim))
            X = x_mean + sigma * (Y @ C_sqrt_inv.T)

            # Evaluate offspring
            F = []
            for i in range(lam):
                if calls_made >= budget or time.time() >= deadline:
                    break
                xi, fi = safe_call(X[i])
                F.append(fi)

            if len(F) < mu:
                return x_mean, sigma, C, pc, ps, False

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx[:mu]]
            F_sorted = np.array(F[idx[:mu]])

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            # Update mean
            x_new_mean = np.sum(w[:, None] * X_sorted, axis=0)

            # Update step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x_new_mean - x_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam)))) < (1.4 + 2 / (dim + 1)) * chi_n

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new_mean - x_mean) / sigma
            delta_h = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + delta_h) * C
            C += c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            C += cmu * np.sum(w[:, None, None] * np.outer(X_sorted - x_mean, X_sorted - x_mean), axis=0) / sigma ** 2

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_c = np.maximum(eigvals_c, 1e-12)
            C = eigvecs_c @ np.diag(eigvals_c) @ eigvecs_c.T

            return x_new_mean, sigma, C, pc, ps, True

        # Run CMA-ES
        converged = False
        for gen in range(200):  # Max generations, but limited by budget/time
            if calls_made >= budget or time.time() >= deadline:
                break

            x_mean, sigma, C, pc, ps, success = cma_es_step(x_mean, sigma, C, pc, ps)
            if not success:
                break

            # Check convergence
            if sigma < 1e-6:
                break

            # If we've made good progress, we can stop early to save time
            if calls_made > budget * 0.8 and best_f < 1.0:
                break

        # If we still have budget, do a local refinement around best
        if calls_made < budget and time.time() < deadline:
            if best_x is not None:
                # Nelder-Mead style local refinement
                x = best_x.copy()
                fx = best_f
                simplex = [x.copy()]
                for i in range(dim):
                    xi = x.copy()
                    xi[i] += 0.5 * (1.0 if xi[i] < 4.5 else -1.0)
                    simplex.append(xi)

                for _ in range(min(50, budget - calls_made)):
                    if time.time() >= deadline:
                        break
                    # Evaluate simplex
                    F = [best_f]
                    for i in range(1, dim + 1):
                        if calls_made >= budget:
                            break
                        _, fi = safe_call(simplex[i])
                        F.append(fi)
                    if len(F) < dim + 1:
                        break

                    # Find worst and best
                    worst_idx = np.argmax(F)
                    best_idx = np.argmin(F)

                    # Centroid of all but worst
                    centroid = np.mean([simplex[i] for i in range(dim + 1) if i != worst_idx], axis=0)

                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[worst_idx])
                    x_ref = np.clip(x_ref, -5, 5)
                    if calls_made >= budget:
                        break
                    _, f_ref = safe_call(x_ref)

                    if f_ref < F[best_idx]:
                        # Expansion
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - simplex[worst_idx])
                        x_exp = np.clip(x_exp, -5, 5)
                        if calls_made < budget:
                            _, f_exp = safe_call(x_exp)
                            if f_exp < f_ref:
                                simplex[worst_idx] = x_exp
                                F[worst_idx] = f_exp
                            else:
                                simplex[worst_idx] = x_ref
                                F[worst_idx] = f_ref
                        else:
                            simplex[worst_idx] = x_ref
                            F[worst_idx] = f_ref
                    elif f_ref < F[worst_idx]:
                        simplex[worst_idx] = x_ref
                        F[worst_idx] = f_ref
                    else:
                        # Contraction
                        beta = 0.5
                        x_con = centroid + beta * (simplex[worst_idx] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        if calls_made < budget:
                            _, f_con = safe_call(x_con)
                            if f_con < F[worst_idx]:
                                simplex[worst_idx] = x_con
                                F[worst_idx] = f_con
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    if i == best_idx:
                                        continue
                                    if calls_made >= budget:
                                        break
                                    simplex[i] = simplex[best_idx] + 0.5 * (simplex[i] - simplex[best_idx])
                                    _, fi = safe_call(simplex[i])
                                    F[i] = fi

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
