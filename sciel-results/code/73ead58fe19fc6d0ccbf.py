"""Multi-start CMA-ES with Halton initialization and Nelder-Mead polish for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with Halton initialization and Nelder-Mead polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Initialization using Halton sequence
        n_init = 25
        init_results = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.05, 0.05, dim)
            point = np.clip(point, -5, 5)

            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        # Sort by fitness
        init_results.sort(key=lambda x: x[1])

        # Phase 2: CMA-ES from top starts
        K = min(3, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(50, remaining_budget // K)

        for x0, f0 in top_starts:
            if time.time() > deadline or calls_made >= budget:
                break

            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))  # lambda
            pop_size = max(pop_size, 6)
            sigma = 1.0
            c1 = (4 + np.log(dim)) / ((dim + 4) ** 2 + 4)
            cmu = min(0.2, (2 - c1 - 1 / dim) / ((dim + 1.3) ** 2 + 6))
            cc = (4 + dim / dim) / (dim + 4)
            cs = (mu_eff + 1 + 2 / (dim + 1 / mu_eff)) / (dim + mu_eff + 5)
            mu_eff = (sum(np.sort(np.arange(pop_size))[:pop_size // 2]) ** 2) / (sum(np.arange(pop_size)) ** 2)
            # Actually compute mu_eff properly
            weights = np.zeros(pop_size)
            weights[:pop_size // 2] = np.log(pop_size / 2 + 0.5) - np.log(1 + np.arange(pop_size // 2))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights[:pop_size // 2] ** 2)

            c1 = (4 + np.log(dim)) / ((dim + 4) ** 2 + 4)
            cmu = min(0.2, (2 - c1 - 1 / dim) / ((dim + 1.3) ** 2 + 6))
            cc = (4 + dim / dim) / (dim + 4)
            cs = (mu_eff + 1 + 2 / (dim + 1 / mu_eff)) / (dim + mu_eff + 5)

            ps = np.zeros(dim)
            C = np.eye(dim)
            Cs = np.eye(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x = x0.copy()
            fx = f0
            es_calls = 0
            max_es_calls = budget_per_start

            while es_calls < max_es_calls:
                if time.time() > deadline or calls_made >= budget:
                    break

                # Generate population
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                y = np.array([rng.standard_normal(dim) for _ in range(pop_size)])
                z = y @ (B * D)  # z_i = B * D * y_i
                x_pop = np.array([x + sigma * z[i] for i in range(pop_size)])
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate population
                f_pop = []
                for xi in x_pop:
                    if time.time() > deadline or calls_made >= budget:
                        break
                    fi = evaluate(xi)
                    f_pop.append(fi)
                    es_calls += 1

                if len(f_pop) < pop_size:
                    break

                # Sort by fitness
                order = np.argsort(f_pop)
                x_pop = x_pop[order]
                f_pop = f_pop[order]

                # Update best
                if f_pop[0] < best_f:
                    best_f = f_pop[0]
                    best_x = x_pop[0].copy()

                # Update mean
                y_w = np.array([B.T @ (D * (B.T @ (x_pop[i] - x) / sigma)) for i in range(pop_size // 2)])
                x_new = x + sigma * np.sum(weights[:pop_size // 2, np.newaxis] * y_w, axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * B.T @ (D * (B.T @ (x_pop[0] - x) / sigma))
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (es_calls + 1))) < (1 + 2 / (dim + 1)) * chiN else 0
                Cs = (1 - cs) * Cs + hsig * cs * (2 - cs) * np.outer(ps, ps)

                # Update covariance matrix
                C = (1 + c1 * hsig * (cc - c1) + cmu) * C
                C -= (c1 * hsig * (cc - c1) + cmu) * np.eye(dim)
                C += c1 * np.outer(ps, ps)
                y_w_all = np.array([B.T @ (D * (B.T @ (x_pop[i] - x) / sigma)) for i in range(pop_size // 2)])
                C += cmu * np.sum(weights[:pop_size // 2, np.newaxis] * np.outer(y_w_all, y_w_all.T), axis=0)

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-6, 50.0)

                x = x_new

                # Check for convergence
                if np.max(np.abs(C)) < 1e-10:
                    break

            # Nelder-Mead refinement after CMA-ES
            if time.time() > deadline or calls_made >= budget:
                continue

            x_nm = best_x.copy()
            f_nm = best_f

            delta = 0.2
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [f_nm]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i + 1])
                simplex_f.append(f_new)

            if len(simplex_f) < dim + 1:
                continue

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = min(30, budget - calls_made)
            no_improve_count = 0

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break

                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                max_dist = 0.0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-8:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    no_improve_count = 0
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    no_improve_count = 0
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        no_improve_count = 0
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve_count += 1

                if no_improve_count > 4:
                    break

        # Phase 3: Final polish on global best
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.03
            max_polish = min(40, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 1.15
                else:
                    sigma_p *= 0.85

                sigma_p = np.clip(sigma_p, 0.001, 0.5)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
