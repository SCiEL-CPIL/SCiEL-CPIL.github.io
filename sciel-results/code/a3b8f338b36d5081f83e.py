"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MO-CMA-ES with non-dominated archive for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        span = hi - lo

        # Global non-dominated archive
        archive_F = []
        archive_X = []

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def add_to_archive(f, x):
            # Check if f is dominated by any existing point
            for af in archive_F:
                if dominates(af, f):
                    return False
            # Remove points dominated by f
            to_remove = []
            for i, af in enumerate(archive_F):
                if dominates(f, af):
                    to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):
                archive_F.pop(idx)
                archive_X.pop(idx)
            archive_F.append(f)
            archive_X.append(x)
            return True

        # Hypervolume contribution (simplified for 3D)
        def hv_contribution(f, ref):
            """Compute hypervolume contribution of f w.r.t. ref and other archive points."""
            if np.any(f >= ref):
                return 0.0
            # Volume of box from f to ref
            vol = 1.0
            for i in range(n_obj):
                vol *= (ref[i] - f[i])
            # Subtract overlaps with other archive points (approximate)
            return vol

        # MO-CMA-ES parameters
        pop_size = 20
        lambda_ = pop_size
        mu = pop_size // 2
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_cov = (2 + 2 * np.sqrt(mu / dim)) / (dim + 2 + np.sqrt(mu / dim))
        c_c = (4 + dim / 4) / (dim + 4 + 2 * mu)
        chi_dim = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES state
        sigma = 0.5
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        ps = 0.0
        hsig = 0.0

        # Decomposition weights for WFG7 (3 objectives)
        n_weights = 10
        weights = []
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = n_weights - i - j - 1
                w = np.array([i, j, k], dtype=float) / (n_weights - 1)
                weights.append(w)
        weights = np.array(weights)
        # Normalize
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-10)

        used = 0
        best_x = None

        # Initial population
        pop_X = lo + span * rng.random((pop_size, dim))
        pop_F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            pop_F[i] = problem(pop_X[i])
            used += 1
            add_to_archive(pop_F[i].copy(), pop_X[i].copy())

        # Set initial mean from best archive point
        if len(archive_X) > 0:
            idx = np.argmin(np.sum(archive_F, axis=1))
            mean = archive_X[idx].copy()
        else:
            mean = (lo + hi) / 2

        # CMA-ES main loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))

            for i in range(pop_size):
                # Sample from multivariate normal with covariance C
                z = rng.standard_normal(dim)
                # Transform: B @ (D * z)
                step = B @ (D * z)
                new_x = mean + sigma * step
                new_x = np.clip(new_x, lo, hi)
                offspring_X[i] = new_x
                offspring_F[i] = problem(new_x)
                used += 1
                add_to_archive(offspring_F[i].copy(), new_x.copy())

            # Selection: pick non-dominated points, then by hypervolume
            # Simple selection: take first mu non-dominated from offspring
            nd_idx = []
            for i in range(pop_size):
                is_nd = True
                for j in range(pop_size):
                    if i != j and dominates(offspring_F[j], offspring_F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)

            if len(nd_idx) < mu:
                # Fill with best dominated points by scalarized objective
                scalar = []
                for i in range(pop_size):
                    if i not in nd_idx:
                        s = np.sum(offspring_F[i] * weights[0])
                        scalar.append((s, i))
                scalar.sort()
                nd_idx += [idx for _, idx in scalar[:mu - len(nd_idx)]]

            nd_idx = nd_idx[:mu]
            nd_idx.sort()

            # Update mean
            new_mean = np.mean(offspring_X[nd_idx], axis=0)

            # Update covariance and step size
            if mu > 1:
                # Weighted sum of outer products
                w = np.log(mu + 0.5) - np.arange(1, mu + 1)
                w = w / np.sum(w)
                mu_eff = 1 / np.sum(w ** 2)

                # Delta mean
                delta_m = (new_mean - mean) / sigma

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_m
                ps = np.linalg.norm(p_sigma)

                # Update p_c
                hsig = 1.0 if ps < (1 + 0.5 * c_sigma) * chi_dim else 0.0
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_m

                # Update C
                delta_C = (1 / (1 - (1 - c_cov) ** (2 * pop_size))) * (
                    (c_cov + (2 - c_cov) * (1 - hsig) / mu_eff) * np.outer(p_c, p_c) - c_cov * C
                )
                # Add rank-mu update
                for i in range(mu):
                    y_i = (offspring_X[nd_idx[i]] - mean) / sigma
                    delta_C += w[i] * np.outer(y_i, y_i)
                C = C + delta_C

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Eigenvalue decomposition for B and D
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    pass

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (ps / chi_dim - 1))
                sigma = np.clip(sigma, 1e-8, 2.0)

            mean = new_mean

            # Periodic restart if stuck
            if used % 50 == 0 and len(archive_X) > 5:
                # Pick a diverse archive point
                pick_idx = rng.integers(len(archive_X))
                mean = archive_X[pick_idx].copy()
                sigma = 0.3
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        return best_x if best_x is not None else (lo + hi) / 2
# EVOLVE-BLOCK-END
