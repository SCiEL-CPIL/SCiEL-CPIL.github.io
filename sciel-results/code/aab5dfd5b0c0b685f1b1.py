"""Optimizer for BBOB-style f23 (Katsuura) using LHS + CMA-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS exploration + CMA-ES exploitation with stagnation-triggered restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counting_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            x_arr = np.clip(np.asarray(x, dtype=float).flatten(), -5.0, 5.0)
            call_count[0] += 1
            fx = func(x_arr)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x_arr.copy()
            return fx

        def lhs_sample(n_samples, dim, rng):
            """Latin Hypercube Sampling in [-5,5]^dim."""
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = rng.permutation(n_samples)
                points[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples * 10.0 - 5.0
            return points

        # Phase 1: LHS exploration (~25% of budget)
        n_lhs = max(15, min(60, budget // 4))
        lhs_points = lhs_sample(n_lhs, dim, rng)
        for x0 in lhs_points:
            if call_count[0] >= budget or time.time() > deadline:
                break
            counting_func(x0)

        if best_x[0] is None or call_count[0] >= budget or time.time() > deadline:
            return best_x[0], best_f[0]

        # Phase 2: CMA-ES with restarts
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda = 4 + 3*ln(n)
        sigma = 2.0  # initial step size
        max_iterations = budget - call_count[0]

        def run_cmaes(x0, sigma0, max_iter, deadline):
            """Run CMA-ES from x0 with initial sigma0, return best_x, best_f, calls_used."""
            x = x0.copy()
            sigma = sigma0
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2  # number of parents

            # CMA-ES state
            C = np.eye(dim)  # covariance matrix
            C_sqrt_inv = np.eye(dim)  # inverse square root of C
            ps = np.zeros(dim)  # path for sigma adaptation
            pc = np.zeros(dim)  # path for covariance adaptation
            cs = 1.0 / (np.sqrt(pop_size) + 3)  # sigma adaptation rate
            ds = 1.0 + 2 * max(0, (np.sqrt((pop_size - 1) / dim + 1) - 1)) / (dim + 2)
            cmu = 2 / ((dim + 1.3) ** 2 + pop_size)  # covariance adaptation rate
            c1 = (2 - cmu) / ((dim + 1) ** 2 + pop_size)
            cmu = 2 / ((dim + 1.3) ** 2 + pop_size)

            best_local_x = x.copy()
            best_local_f = np.inf
            calls_used = 0
            no_improve = 0
            patience = 50  # restart after 50 iterations without improvement

            for it in range(max_iter):
                if call_count[0] >= budget or time.time() > deadline:
                    break

                # Generate population
                # Compute C_sqrt_inv update (simplified: use eigenvalue decomposition periodically)
                if it % 10 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_sqrt_inv = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

                # Generate candidates
                candidates = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    candidates[i] = x + sigma * (C_sqrt_inv @ z)
                    candidates[i] = np.clip(candidates[i], -5.0, 5.0)

                # Evaluate candidates
                fvals = np.zeros(pop_size)
                for i in range(pop_size):
                    if call_count[0] >= budget or time.time() > deadline:
                        fvals[i] = np.inf
                        break
                    fvals[i] = counting_func(candidates[i])
                    calls_used += 1

                # Sort by fitness
                indices = np.argsort(fvals)
                x_sorted = candidates[indices]
                f_sorted = fvals[indices]

                # Update best
                if f_sorted[0] < best_local_f:
                    best_local_f = f_sorted[0]
                    best_local_x = x_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Select parents
                parents = x_sorted[:mu]
                weights = np.zeros(mu)
                w_max = np.log(mu) - np.log(1)
                for i in range(mu):
                    weights[i] = np.log(mu + 0.5) - np.log(i + 1)
                weights /= np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)

                # Update mean
                old_x = x.copy()
                x = np.sum(weights[:, None] * parents, axis=0)

                # Update path for sigma adaptation
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (C_sqrt_inv.T @ (x - old_x) / sigma)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mueff):
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))

                # Update path for covariance adaptation
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mueff) else 0
                pc = (1 - c1) * pc + c1 * hsig * (x - old_x) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                rank_one = np.outer(x - old_x, x - old_x)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = C_sqrt_inv.T @ (parents[i] - old_x) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * np.eye(dim)) + cmu * rank_mu

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

                # Check for stagnation - restart if no improvement for patience iterations
                if no_improve >= patience:
                    break

            return best_local_x, best_local_f, calls_used

        # Run CMA-ES from best LHS point
        x0 = best_x[0].copy()
        sigma0 = 2.0
        remaining = budget - call_count[0]

        best_x[0], best_f[0], _ = run_cmaes(x0, sigma0, remaining, deadline)

        # Phase 3: If budget remains, do a few more restarts from perturbed best
        remaining = budget - call_count[0]
        if remaining > 20 and best_x[0] is not None and time.time() < deadline:
            for restart in range(3):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                # Perturb best point
                x_perturbed = best_x[0].copy() + 0.5 * rng.standard_normal(dim)
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                counting_func(x_perturbed)

                remaining = budget - call_count[0]
                if remaining < 10:
                    break

                bx, bf, _ = run_cmaes(x_perturbed, 1.0, remaining, deadline)
                if bf < best_f[0]:
                    best_x[0], best_f[0] = bx, bf

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
