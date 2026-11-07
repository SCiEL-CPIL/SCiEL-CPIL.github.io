"""CMA-ES optimizer for BBOB f9 (rotated Rosenbrock) with multi-start and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with early AOCC credit."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best value found so far
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_f, best_x, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x) != dim:
                x = x[:dim] if len(x) > dim else np.pad(x, (0, dim - len(x)))
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse global search with multiple diverse starting points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        n_starts = 8
        starts = []
        # Include origin and a few structured points
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)
        starts.append(rng.uniform(-3, 3, dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-5, 5, dim))
        starts.append(rng.uniform(-1, 1, dim) * 4.0)
        starts.append(rng.uniform(-2, 2, dim))

        # Evaluate all starting points
        start_results = []
        for s in starts:
            if call_count >= budget or time.time() > deadline:
                break
            fs = evaluate(s)
            start_results.append((fs, s.copy()))

        if not start_results:
            # Fallback: at least one evaluation
            s0 = np.zeros(dim)
            evaluate(s0)
            return best_x, best_f

        # Sort by function value
        start_results.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best few starting points
        # Use CMA-ES with a population size appropriate for dim=5
        # Budget allocation: use remaining budget for CMA-ES

        def cma_es(start_x, start_f, max_calls, rng_cma):
            """Simple CMA-ES implementation."""
            nonlocal call_count, best_f, best_x

            if max_calls <= 0:
                return start_x, start_f

            dim_c = dim
            # Population size
            lambda_c = max(4, int(4 + 3 * np.log(dim_c)))
            # Initial sigma
            sigma_c = 1.0
            x_mean = start_x.copy()
            f_mean = start_f

            # Covariance matrix
            C = np.eye(dim_c)
            # Inverse square root of C
            C_sqrt_inv = np.eye(dim_c)
            # Eigenvalues
            eigvals = np.ones(dim_c)

            # Strategy parameters
            mu = lambda_c // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim_c + mu_eff + 5)
            c_c = (4 + mu_eff / dim_c) / (dim_c + 4 + 2 * mu_eff / dim_c)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim_c + 1)) - 1) + c_sigma

            p_sigma = np.zeros(dim_c)
            p_c = np.zeros(dim_c)
            chi_n = np.sqrt(dim_c) * (1 - 1 / (4 * dim_c) + 1 / (21 * dim_c ** 2))

            archive = []  # (f, x) pairs for this CMA-ES run

            for gen in range(max_calls // lambda_c + 1):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                for i in range(lambda_c):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng_c.standard_normal(dim_c)
                    # Transform using C matrix
                    # For simplicity, use C_sqrt_inv @ z approximation
                    step = C_sqrt_inv @ z
                    x_child = x_mean + sigma_c * step
                    x_child = np.clip(x_child, -5, 5)
                    f_child = evaluate(x_child)
                    archive.append((f_child, x_child.copy()))

                if len(archive) < lambda_c:
                    break

                # Sort by function value
                archive.sort(key=lambda t: t[0])
                selected = archive[:mu]

                # Update mean
                x_new_mean = np.zeros(dim_c)
                for w, (f, x) in zip(weights, selected):
                    x_new_mean += w * x
                x_mean = x_new_mean

                # Update covariance
                # Compute weighted sum of outer products
                diff = np.array([x - x_mean for f, x in selected])
                C_new = np.zeros((dim_c, dim_c))
                for w, d in zip(weights, diff):
                    C_new += w * np.outer(d, d)
                C_new /= sigma_c ** 2

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma_c * (
                    x_mean - x_old_mean if 'x_old_mean' in dir() else np.zeros(dim_c)
                )
                # Actually, we need to track the old mean
                # Let me fix this - restructure

                # Simpler approach: just track progress
                if archive[0][0] < f_mean:
                    f_mean = archive[0][0]
                    sigma_c *= 1.1
                else:
                    sigma_c *= 0.95

                sigma_c = max(sigma_c, 1e-8)
                sigma_c = min(sigma_c, 10.0)

            return x_mean, f_mean

        # Instead of the buggy CMA-ES above, use a cleaner implementation
        def cma_es_clean(start_x, start_f, max_calls, rng_cma):
            """Cleaner CMA-ES for dim=5."""
            nonlocal call_count, best_f, best_x

            if max_calls <= 0:
                return start_x, start_f

            dim_c = dim
            lambda_c = max(4, int(4 + 3 * np.log(dim_c)))
            sigma_c = 0.5
            x_mean = start_x.copy()

            C = np.eye(dim_c)
            C_eigvals = np.ones(dim_c)
            C_eigvecs = np.eye(dim_c)

            mu = max(1, lambda_c // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim_c + mu_eff + 5)
            c_c = (4 + mu_eff / dim_c) / (dim_c + 4 + 2 * mu_eff / dim_c)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim_c + 1)) - 1) + c_sigma

            p_sigma = np.zeros(dim_c)
            p_c = np.zeros(dim_c)
            chi_n = np.sqrt(dim_c) * (1 - 1 / (4 * dim_c) + 1 / (21 * dim_c ** 2))

            x_old_mean = x_mean.copy()
            f_best_local = start_f

            for gen in range(max_calls // lambda_c + 2):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_c):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng_cma.standard_normal(dim_c)
                    # Use eigenvalue decomposition for transformation
                    # C_sqrt = C_eigvecs @ diag(sqrt(C_eigvals)) @ C_eigvecs.T
                    # step = C_sqrt @ z
                    step = C_eigvecs @ (np.sqrt(C_eigvals) * (C_eigvecs.T @ z))
                    x_child = x_mean + sigma_c * step
                    x_child = np.clip(x_child, -5, 5)
                    f_child = evaluate(x_child)
                    offspring.append((f_child, x_child.copy()))

                if len(offspring) < mu:
                    break

                # Sort
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]

                # Update mean
                x_new_mean = np.zeros(dim_c)
                for w, (f, x) in zip(weights, selected):
                    x_new_mean += w * x
                x_old_mean = x_mean.copy()
                x_mean = x_new_mean

                # Update path accumulators
                p_sigma = (1 - c_sigma) * p_sigma
                diff_norm = np.linalg.norm(x_mean - x_old_mean)
                if diff_norm > 0:
                    p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_old_mean) / (sigma_c * chi_n)

                p_c = (1 - c_c) * p_c
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.5 + 1 / mu_eff) * chi_n:
                    p_c += np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old_mean) / sigma_c

                # Update covariance
                C = (1 + c_c * (2 - c_c) - c_sigma) * C
                C += c_sigma * (2 - c_sigma) * (p_sigma * (p_sigma + 1 / c_sigma * p_c) / chi_n ** 2)
                # Also add rank-mu update
                for w, (f, x) in zip(weights, selected):
                    diff = (x - x_old_mean) / sigma_c
                    C += w * np.outer(diff, diff)

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_eigvals = eigvals
                C_eigvecs = eigvecs
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Adapt sigma
                sigma_c *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma_c = max(sigma_c, 1e-10)
                sigma_c = min(sigma_c, 50.0)

                # Track best
                if offspring[0][0] < f_best_local:
                    f_best_local = offspring[0][0]

            return x_mean, f_best_local

        # Allocate remaining budget to CMA-ES from top starting points
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Use CMA-ES from the best 2-3 starting points
        n_cma = min(3, len(start_results))
        calls_per_cma = remaining_budget // n_cma

        for idx in range(n_cma):
            if call_count >= budget or time.time() > deadline:
                break
            fs, xs = start_results[idx]
            rng_cma = np.random.default_rng(100 + idx)
            _, _ = cma_es_clean(xs, fs, calls_per_cma, rng_cma)

        # Phase 3: Final local refinement from the best point found
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if best_x is not None:
                # Simple (1+1)-ES refinement
                x_refine = best_x.copy()
                f_refine = best_f
                sigma_refine = 0.1
                rng_refine = np.random.default_rng(200)
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    y = np.clip(x_refine + sigma_refine * rng_refine.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < f_refine:
                        x_refine, f_refine = y, fy
                        sigma_refine *= 1.2
                    else:
                        sigma_refine *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
