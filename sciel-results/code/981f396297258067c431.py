"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restart CMA-ES with Nelder-Mead polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, budget_left, rng, func):
            nonlocal best_x, best_f, total_calls

            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fs = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.25 if x0[i] < 4.75 else -0.25
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fs.append(fi)

            if len(simplex) < 2:
                return x0, f0

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iters = min(budget_left, 200)
            for _ in range(max_iters):
                if time.time() > deadline or total_calls >= budget:
                    break

                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Check convergence
                if len(fs) > 1 and abs(fs[0] - fs[-1]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if fi is None:
                            break

                # Update best
                min_idx = np.argmin(fs)
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # CMA-ES with restarts
        def cma_es_with_restarts(x0, sigma0, budget_left, rng, func):
            nonlocal best_x, best_f, total_calls

            n = dim
            max_restarts = 5
            restart_threshold = 50  # iterations without improvement before restart

            for restart in range(max_restarts):
                if time.time() > deadline or total_calls >= budget:
                    break

                # CMA-ES parameters
                lam = max(4, int(4 + 3 * np.log(n)))
                mu = lam // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= np.sum(w)
                mueff = 1.0 / np.sum(w ** 2)
                cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                cs = (mueff + 2) / (n + mueff + 5)
                c1 = (2 - cc + 2 * mueff / (n - 2)) / ((n + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
                chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

                x_mean = np.array(x0, dtype=float)
                sigma = sigma0
                B = np.eye(n)
                D = np.ones(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                evals = 0
                best_local_x = x_mean.copy()
                best_local_f = safe_call(x_mean)
                if best_local_f is None:
                    break
                if best_local_f < best_f:
                    best_f = best_local_f
                    best_x = best_local_x.copy()

                no_improve_count = 0
                prev_best = best_local_f

                while evals < budget_left:
                    # Check deadline
                    if time.time() > deadline:
                        break
                    # Check budget
                    if total_calls >= budget:
                        break

                    # Generate lambda offspring
                    offspring = []
                    for i in range(lam):
                        if time.time() > deadline or total_calls >= budget:
                            break
                        z = rng.standard_normal(n)
                        y = x_mean + sigma * (B @ (D * z))
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None:
                            offspring.append((y, fy))
                            evals += 1
                            if fy < best_local_f:
                                best_local_f = fy
                                best_local_x = y.copy()
                                if fy < best_f:
                                    best_f = fy
                                    best_x = y.copy()

                    if len(offspring) < 2:
                        break

                    # Sort by fitness
                    offspring.sort(key=lambda t: t[1])
                    offspring = offspring[:mu]

                    # Update mean
                    new_mean = np.sum(np.array([w[i] * offspring[i][0] for i in range(mu)]), axis=0)

                    # Update covariance matrix
                    y_inv = [B.T @ (D * (offspring[i][0] - x_mean) / sigma) for i in range(mu)]
                    delta = (new_mean - x_mean) / sigma
                    pc = (1 - cc) * pc + cc * delta
                    hsig = (np.sqrt((1 - (1 - cc) ** (evals + 1)) / (1 - (1 - cc) ** (evals + lam))) *
                            (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (evals + 1)) / chi_n < 1.4 + 2 / (n + 1)))
                    ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * delta

                    # Recompute C from scratch to avoid accumulation errors
                    C = np.eye(n)
                    C = C + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(
                        [w[i] * np.outer(y_inv[i], y_inv[i]) for i in range(mu)], axis=0
                    ) - (c1 + cmu) * C
                    # Ensure positive semi-definite
                    C = (C + C.T) / 2
                    eigenvalues, B = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)

                    # Update sigma
                    sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 1e10)

                    x_mean = new_mean
                    x_mean = np.clip(x_mean, -5, 5)

                    # Check for stagnation
                    if best_local_f < prev_best - 1e-10:
                        no_improve_count = 0
                        prev_best = best_local_f
                    else:
                        no_improve_count += 1

                    if no_improve_count > restart_threshold:
                        # Stagnation detected - break to restart
                        break

                # If we used significant budget and have good result, stop restarting
                if evals > budget_left * 0.7:
                    break

                # Prepare for next restart: start from best, reduce sigma
                x0 = best_local_x.copy()
                sigma0 = max(sigma * 0.5, 1e-4)

            return best_local_x, best_local_f

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # First try: center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        budget_per_run = max(80, budget // 3)

        # Run CMA-ES with restarts from center
        x_res, f_res = cma_es_with_restarts(x0, sigma0, budget_per_run, rng, func)

        # If we still have budget, try a few random starts
        if total_calls < budget and time.time() < deadline:
            for attempt in range(2):
                if total_calls >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                sigma_rand = 1.5
                remaining = budget - total_calls
                if remaining < 50:
                    break
                x_res2, f_res2 = cma_es_with_restarts(x_rand, sigma_rand, min(remaining, budget_per_run // 2), rng, func)

        # Final Nelder-Mead polish on the best point
        if total_calls < budget and time.time() < deadline and best_x is not None:
            nm_budget = min(budget - total_calls, 150)
            if nm_budget > 10:
                x_nm, f_nm = nelder_mead(best_x.copy(), best_f, nm_budget, rng, func)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
