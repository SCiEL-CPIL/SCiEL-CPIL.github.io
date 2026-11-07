"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES style optimizer with time guard.
    Uses a population-based approach with covariance adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        call_count = 0

        def _check_deadline():
            return time.time() > deadline

        def _eval(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if _check_deadline():
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Multi-start local search using Nelder-Mead style or simple hill climbing
        # Phase 2: CMA-ES with population

        # First, do a quick multi-start with simple hill climbing to find a good initial region
        n_starts = 10
        starts = []
        for i in range(n_starts):
            if call_count >= budget or _check_deadline():
                break
            x0 = rng.uniform(-4, 4, dim)
            fx0 = _eval(x0)
            starts.append((fx0, x0.copy()))

        # Sort starts by fitness
        starts.sort(key=lambda t: t[0])

        # For the best start, do a local refinement with (1+1)-ES
        if starts:
            fx_best, x_best = starts[0]
            # Refine the best start
            sigma = 1.0
            x = x_best.copy()
            fx = fx_best
            for _ in range(min(50, budget - call_count)):
                if call_count >= budget or _check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = _eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if sigma < 1e-6:
                    break

        # Now run CMA-ES from the best found point
        if call_count >= budget or _check_deadline():
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))
        pop_size = min(pop_size, 20)
        sigma = 1.0
        x_mean = x.copy() if x is not None else rng.uniform(-4, 4, dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = 1 / (pop_size + 4)
        c_c = (4 + pop_size / dim) / (pop_size + 4 + 4 / dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_cov = (2 + 2 / (mu_eff + 1)) / (dim + 2 + 2 / (mu_eff + 1))
        d_cov = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_cov

        def _matvec_sqrt_m(C, v):
            """Compute C^{-1/2} v via Cholesky of C (assuming C is PD)."""
            try:
                L = np.linalg.cholesky(C)
                # Solve L L^T v = w => L^T w = v => w = L^{-T} v
                # Actually we want C^{-1/2} v. Use Cholesky: C = L L^T
                # C^{-1/2} = L^{-T} L^{-1} ... this is getting complex.
                # Simpler: use eigendecomposition for small dim
                pass
            except:
                pass
            # Use eigendecomposition for robustness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_inv_sqrt @ v
            except:
                return v

        for gen in range(1000):
            if call_count >= budget or _check_deadline():
                break

            # Generate population
            Y = np.zeros((pop_size, dim))
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if call_count >= budget or _check_deadline():
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # C^{1/2} z
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    step = C_sqrt @ z
                except:
                    step = z
                y = np.clip(x_mean + sigma * step, -5, 5)
                Y[i] = y
                F[i] = _eval(y)

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * Y[:mu], axis=0)
            # Compute evolution path for covariance
            y_inv = np.zeros(dim)
            for i in range(mu):
                y_inv += weights[i] * (Y[i] - x_mean) / sigma

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_inv
            hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) \
                   < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv

            # Update sigma
            sigma *= np.exp(c_sigma / d_cov * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            delta_h = (1.0 if hsig else 0.0) - (1 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (Y[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            C = (1 - c_cov * (1 + d_cov - c_cov)) * C + c_cov * (rank_one + delta_h * rank_mu)

            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new_mean

            # If no improvement in best, consider restarting or adjusting
            if gen % 20 == 0 and best_f == np.inf:
                # This shouldn't happen since we evaluated starts
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
