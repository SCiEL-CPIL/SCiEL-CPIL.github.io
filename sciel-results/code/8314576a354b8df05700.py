"""Combined optimizer: Rastrigin-aware initialization + CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware multi-start optimizer with CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        def check_time():
            return calls_made < budget and (time.time() - t_start) < deadline - 5

        def cma_es(x0, fx0, sigma0=1.0, max_evals=80):
            """CMA-ES for local refinement around a candidate minimum."""
            nonlocal best_f, best_x
            n = dim
            pop_size = max(4, int(4 + 3 * np.log(n)))

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # State
            mean = x0.copy().astype(float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (20 * n ** 2))

            best_local_x = mean.copy()
            best_local_f = fx0

            for gen in range(max_evals // pop_size + 2):
                if not check_time():
                    break

                # Generate population
                Y = []
                for i in range(pop_size):
                    if not check_time():
                        break
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x_trial = mean + sigma * y
                    x_trial = np.clip(x_trial, -5, 5)
                    Y.append((x_trial, z, y))

                # Evaluate
                fvals = []
                for x_trial, z, y in Y:
                    f_trial = safe_func(x_trial)
                    fvals.append(f_trial)
                    if f_trial < best_local_f:
                        best_local_f = f_trial
                        best_local_x = x_trial.copy()
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()

                if not check_time():
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                Y_sorted = [Y[i] for i in order]
                fvals_sorted = [fvals[i] for i in order]

                # Update mean (weighted recombination)
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * Y_sorted[i][0]
                mean = new_mean

                # Update step size
                old_sigma = sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(mean - (mean - sigma * B @ (D * np.mean([Y_sorted[i][1] for i in range(mu)], axis=0)))) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update evolution paths
                y_mean = (mean - (mean - sigma * B @ (D * np.mean([Y_sorted[i][1] for i in range(mu)], axis=0)))) / (sigma + 1e-32)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

                # Update covariance matrix
                delta = h_sigma * (1 - c_c) - (1 - c_c)
                delta = h_sigma * (1 - c_c)
                outer = np.outer(pc, pc)
                for i in range(mu):
                    y_i = Y_sorted[i][1]
                    outer += (weights[i] / (c_c * (2 - c_c))) * np.outer(B @ (D * y_i), B @ (D * y_i))
                C = (1 - c_c * (2 - c_c) + c_c * c_c * delta) * (B @ np.diag(D ** 2) @ B.T) + c_c * (2 - c_c) * (outer - (B @ np.diag(D ** 2) @ B.T))

                # Eigendecomposition for stability
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-16, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    pass

                # Check convergence
                if np.max(D) / np.min(D) > 1e12 or sigma < 1e-14:
                    break

            return best_local_x, best_local_f

        # Phase 1: Rastrigin-aware initialization
        # Rastrigin minima are at integer coordinates. Sample integer grid points
        # and nearby offsets to find promising basins.
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Integer grid points (sparse for dim=5: not all 9^5=59049)
        # Use a stratified subset
        for _ in range(40):
            pt = rng.integers(-4, 5, dim).astype(float)
            starts.append(pt)

        # Near-integer points (offsets of 0.3-0.7 from integers)
        for _ in range(20):
            base = rng.integers(-4, 5, dim).astype(float)
            offsets = rng.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dim)
            pt = base + offsets
            pt = np.clip(pt, -5, 5)
            starts.append(pt)

        # Random uniform for diversity
        for _ in range(10):
            pt = rng.uniform(-4.5, 4.5, dim)
            starts.append(pt)

        starts = np.array(starts)

        # Phase 2: Evaluate all starts, keep top candidates
        start_results = []
        for x0 in starts:
            if not check_time():
                break
            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            start_results.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by function value
        start_results.sort(key=lambda t: t[1])

        # Phase 3: CMA-ES refinement on top candidates
        n_refine = min(6, len(start_results))
        for i in range(n_refine):
            if not check_time():
                break
            x0, fx0 = start_results[i]
            remaining = budget - calls_made
            if remaining < 20:
                break
            # Allocate more evals to better candidates
            frac = max(0.1, 1.0 - 0.1 * i)
            max_evals = int(remaining * frac * 0.7)
            max_evals = max(30, min(max_evals, 120))
            x, fx = cma_es(x0, fx0, sigma0=1.5, max_evals=max_evals)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 4: Final polish with very fine CMA-ES on the best point
        if best_x is not None and check_time():
            remaining = budget - calls_made
            if remaining > 30:
                x, fx = cma_es(best_x.copy(), best_f, sigma0=0.3, max_evals=min(60, remaining))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 5: One more round of random restarts if budget remains
        if check_time():
            remaining = budget - calls_made
            n_extra = min(remaining // 25, 4)
            for _ in range(n_extra):
                if not check_time():
                    break
                x_trial = rng.uniform(-5, 5, dim)
                f_trial = safe_func(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    remaining = budget - calls_made
                    if remaining > 30:
                        x, fx = cma_es(x_trial.copy(), f_trial, sigma0=1.0, max_evals=min(50, remaining))
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
