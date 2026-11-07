"""Optimizer using CMA-ES with multi-start and L-BFGS-B polishing for BBOB f8 (Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start and local polishing for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline and call_count[0] < budget

        def cma_es(x0, sigma, max_evals, seed=0):
            """
            CMA-ES implementation for box-constrained optimization.
            Returns (best_x, best_f, num_calls_used)
            """
            rng = np.random.default_rng(seed)
            n = dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            mu = lam // 2  # number of best solutions used for update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size control
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) * (c_c + 1) + 2 / (n + 1.3))
            c_mu = min(1 - c_1, (2 + (mu_eff - 2) / (n + 2)) / ((n + 2) * (c_c + 1) + (mu_eff - 2) / (n + 2)))

            # State
            x_mean = np.array(x0, dtype=float)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma
            inv_sqrt_C = np.eye(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_x_local = x_mean.copy()
            best_f_local = np.inf
            calls_used = 0

            def safe_clip(x):
                return np.clip(x, -5.0, 5.0)

            for gen in range(max_evals // lam + 1):
                if not time_ok():
                    break
                if calls_used >= max_evals:
                    break

                # Generate population
                ys = rng.standard_normal((lam, n))
                xs = np.array([safe_clip(x_mean + sigma * (ys[i] @ inv_sqrt_C)) for i in range(lam)])

                # Evaluate
                fs = []
                for x in xs:
                    if not time_ok() or calls_used >= max_evals:
                        break
                    f = wrapped_func(x)
                    fs.append(f)
                    calls_used += 1
                    if f < best_f_local:
                        best_f_local = f
                        best_x_local = x.copy()

                if len(fs) < lam:
                    # Pad with infinity if we ran out of budget
                    while len(fs) < lam:
                        fs.append(np.inf)

                # Sort
                idx = np.argsort(fs)
                xs_sorted = xs[idx]
                fs_sorted = fs[idx]

                # Update mean
                x_new_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
                x_new_mean = safe_clip(x_new_mean)

                # Update covariance
                y = (x_new_mean - x_mean) / sigma
                pc = (1 - c_c) * pc + c_c * y
                norm_pc = np.linalg.norm(pc)
                hsig = (norm_pc / np.sqrt(1 - (1 - c_c) ** (gen + 1)) / chi_n) < (1 + c_sigma) * 1.4 + 1e-9

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y / chi_n
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                delta_hsig = (1 - hsig) * hsig * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + \
                    c_1 * np.outer(pc, pc) + \
                    c_mu / (sigma ** 2) * np.sum(weights[i] * np.outer(
                        (xs_sorted[i] - x_mean) / sigma,
                        (xs_sorted[i] - x_mean) / sigma
                    ) for i in range(mu))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update inverse sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    inv_sqrt_C = np.eye(n)

                x_mean = x_new_mean

                # Convergence check
                if np.linalg.norm(sigma * inv_sqrt_C, 'fro') < 1e-8:
                    break

            return best_x_local, best_f_local, calls_used

        # Phase 1: Initial sampling to get a rough idea
        rng = np.random.default_rng(42)
        n_init = 15
        for i in range(n_init):
            if not time_ok():
                break
            if i == 0:
                x0 = np.zeros(dim)
            elif i <= 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Multi-start CMA-ES
        # Try CMA-ES from several starting points with different sigma values
        starts = []
        if best_x is not None:
            starts.append((best_x.copy(), 1.0, 0))
            starts.append((best_x + rng.normal(0, 0.5, dim), 0.5, 1))
            starts.append((best_x + rng.normal(0, 1.0, dim), 1.0, 2))
            starts.append((best_x + rng.normal(0, 2.0, dim), 2.0, 3))
        
        # Add some random starts
        for i in range(4):
            x_rand = rng.uniform(-4, 4, dim)
            starts.append((x_rand, 1.5, 100 + i))

        # Shuffle starts to diversify
        rng.shuffle(starts)

        for x0, sigma, seed in starts:
            if not time_ok():
                break
            if call_count[0] >= budget - 20:
                break

            # Limit evals per CMA-ES run
            remaining = budget - call_count[0] - 10
            max_evals = min(remaining, 150)
            if max_evals < 20:
                break

            x_best, f_best, _ = cma_es(x0, sigma, max_evals, seed)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: Local polishing with a simple gradient-free method
        # Use a coordinate descent / Nelder-Mead style refinement
        if best_x is not None and time_ok() and call_count[0] < budget - 5:
            # Simple pattern search / coordinate refinement
            x_cur = best_x.copy()
            f_cur = best_f
            step_size = 0.1
            n_refine = 0
            max_refine = 50

            for _ in range(max_refine):
                if not time_ok() or call_count[0] >= budget - 2:
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        x_try = x_cur.copy()
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, -5, 5)
                        if not time_ok() or call_count[0] >= budget - 2:
                            break
                        f_try = wrapped_func(x_try)
                        if f_try < f_cur - 1e-10:
                            x_cur = x_try
                            f_cur = f_try
                            best_f = f_cur
                            best_x = x_cur.copy()
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                n_refine += 1

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
