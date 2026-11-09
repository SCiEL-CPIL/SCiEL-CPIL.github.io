"""Hybrid CMA-ES + Nelder-Mead ensemble optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES for global exploration with Nelder-Mead ensemble for local refinement.
    CMA-ES handles the rotated landscape efficiently, while multiple Nelder-Mead
    simplexes escape Rastrigin's local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s

        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clip = np.clip(x, -5.0, 5.0)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = x_clip.copy()
            return f_val

        # --- Phase 1: CMA-ES for global exploration ---
        # CMA-ES parameters
        lam = 12  # population size
        mu = lam // 2  # number of points for update

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)

        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Budget allocation: ~70% for CMA-ES, ~30% for local refinement
        cma_budget = int(budget * 0.7)
        cma_calls = 0

        # Initial evaluation
        f0 = safe_call(np.zeros(dim))
        if f0 is None:
            return best_x, best_fx

        # CMA-ES loop
        while cma_calls < cma_budget and calls_made < budget and time.time() < deadline:
            # Compute Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            # Generate population
            samples = []
            f_vals = []

            for i in range(lam):
                if cma_calls + (lam - i) > cma_budget or time.time() > deadline:
                    break
                z = np.random.randn(dim)
                y = x_mean + sigma * (L @ z)
                samples.append(y)

            # Evaluate all samples
            for y in samples:
                f_val = safe_call(y)
                if f_val is None:
                    break
                f_vals.append(f_val)
                cma_calls += 1

            if len(f_vals) < lam:
                break

            # Sort by fitness
            indices = np.argsort(f_vals)
            sorted_samples = [samples[i] for i in indices]
            sorted_f_vals = [f_vals[i] for i in indices]

            # Update mean
            new_x_mean = np.sum(weights[:, np.newaxis] * np.array(sorted_samples[:mu]), axis=0)

            # Update step size sigma
            y_sigma = (new_x_mean - x_mean) / (sigma * np.linalg.norm(L))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_calls + 1)))
                < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
            )
            sigma = sigma * np.exp(
                c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1)
            )

            # Update covariance matrix
            y_c = (new_x_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Rank-one update
            C = (1 + c_1 * (2 - h_sigma) - c_1) * C + c_1 * np.outer(y_c, y_c)

            # Rank-mu update
            delta = h_sigma * c_c * (2 - c_c)
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (np.array(sorted_samples[i]) - x_mean) / sigma
                rank_mu_term += weights[i] * np.outer(y_i, y_i)

            C = C + c_mu * (delta * np.outer(p_c, p_c) + (1 - delta) * np.eye(dim) - rank_mu_term)

            # Symmetrize C
            C = (C + C.T) / 2

            # Update mean
            x_mean = new_x_mean

            # Check if C is still valid
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = np.eye(dim) * max(np.max(eigvals), 1.0)

        # --- Phase 2: Nelder-Mead ensemble for local refinement ---
        nm_budget = budget - calls_made

        # Get top 5 points from CMA-ES final state
        # Use the best point and some diverse points
        start_points = [x_mean.copy()]

        # Add a few random perturbations around the best point
        rng = np.random.default_rng(42)
        for _ in range(4):
            sp = best_x + rng.normal(0, 0.5, dim)
            sp = np.clip(sp, -5, 5)
            start_points.append(sp)

        # Run Nelder-Mead on each starting point
        nm_per_start = max(1, nm_budget // len(start_points))

        for sp in start_points:
            if calls_made >= budget or time.time() > deadline:
                break

            # Initialize simplex around sp
            simplex = [sp.copy()]
            fs = [safe_call(sp)]
            if fs[0] is None:
                break

            for i in range(dim):
                xi = sp.copy()
                xi[i] += 0.1
                simplex.append(xi)
                fi = safe_call(xi)
                if fi is None:
                    break
                fs.append(fi)

            if calls_made >= budget or time.time() > deadline:
                break

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            while calls_made < budget and time.time() < deadline:
                # Sort simplex
                order_s = np.argsort(fs)
                simplex = [simplex[i] for i in order_s]
                fs = [fs[i] for i in order_s]

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
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if calls_made >= budget or time.time() > deadline:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
