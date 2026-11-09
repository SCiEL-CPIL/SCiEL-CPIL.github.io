"""Multi-objective optimizer on WFG7 using MO-CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MO-CMA-ES optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Parameters
        N = self.dim  # population size
        mu = N // 2   # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_cov = (2 + 2 * mu_eff / self.dim) / (self.dim + 1 + mu_eff / self.dim + 2 * mu_eff / self.dim)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma

        # State
        x_mean = lo + (hi - lo) * 0.5
        sigma = 0.5 * np.mean(hi - lo)
        C = np.eye(self.dim)
        C_sqrt_inv = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)

        # Archive
        archive_X = []
        archive_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=np.float64)
            used += 1
            archive_X.append(x_clipped)
            archive_F.append(f)
            return f

        def non_dominated_sort(F):
            """Return indices of non-dominated points in F."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def update_distribution(X, F):
            """Update CMA-ES state based on selected parents."""
            nonlocal x_mean, sigma, C, C_sqrt_inv, p_sigma, p_c

            # Select parents: top mu by non-dominated rank, then crowding
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) >= mu:
                # Use crowding distance to select from non-dominated
                nd_F = F[nd_idx]
                # Simple selection: take first mu non-dominated
                parent_idx = nd_idx[:mu]
            else:
                # Take all non-dominated, then next non-dominated
                parent_idx = list(nd_idx)
                remaining = [i for i in range(len(F)) if i not in nd_idx]
                # Sort remaining by sum of objectives
                sums = F[remaining].sum(axis=1)
                remaining_sorted = np.array(remaining)[np.argsort(sums)]
                parent_idx = parent_idx + list(remaining_sorted[:mu - len(parent_idx)])

            X_parents = X[parent_idx]
            w = weights[:len(parent_idx)]
            w = w / w.sum()

            # Weighted mean
            x_new_mean = w @ X_parents

            # Update path
            old_mean = x_mean.copy()
            x_mean = x_new_mean

            # Normalize
            diff = x_new_mean - old_mean
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-14:
                diff_norm = diff / norm_diff
            else:
                diff_norm = np.zeros(self.dim)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv.T @ diff_norm
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_norm

            # Update sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 5.0 * np.max(hi - lo))

            # Update covariance
            delta = (np.linalg.norm(p_c) ** 2 / c_c * (1 - c_c) - 1)
            rank_mu_term = np.zeros((self.dim, self.dim))
            for i in range(len(parent_idx)):
                y_i = C_sqrt_inv @ (X_parents[i] - old_mean) / sigma
                rank_mu_term += w[i] * np.outer(y_i, y_i)

            C = (1 + c_cov * (c_sigma * delta - 1)) * C + c_cov * rank_mu_term

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Compute inverse square root
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_sqrt_inv = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

        # Initial population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        for i in range(N):
            y = rng.standard_normal(self.dim)
            X[i] = x_mean + sigma * (C_sqrt_inv @ y)
            X[i] = np.clip(X[i], lo, hi)
            F[i] = evaluate(X[i])

        # Evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parents
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) >= mu:
                parent_idx = nd_idx[:mu]
            else:
                parent_idx = list(nd_idx)
                remaining = [i for i in range(N) if i not in nd_idx]
                sums = F[remaining].sum(axis=1)
                remaining_sorted = np.array(remaining)[np.argsort(sums)]
                parent_idx = parent_idx + list(remaining_sorted[:max(0, mu - len(parent_idx))])

            X_parents = X[parent_idx]
            F_parents = F[parent_idx]

            # Update distribution
            update_distribution(X_parents, F_parents)

            # Generate offspring
            for i in range(N):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                y = rng.standard_normal(self.dim)
                x_child = x_mean + sigma * (C_sqrt_inv @ y)
                x_child = np.clip(x_child, lo, hi)
                f_child = evaluate(x_child)

                # Replace worst point
                # Simple strategy: replace point dominated by child, or worst by sum
                dominated_by_child = np.array([
                    np.all(f_child <= f) and np.any(f_child < f)
                    for f in F
                ])
                if np.any(dominated_by_child):
                    # Replace one of the dominated points
                    candidates = np.where(dominated_by_child)[0]
                    idx = candidates[rng.integers(len(candidates))]
                else:
                    idx = int(np.argmax(F.sum(axis=1)))

                X[idx] = x_child
                F[idx] = f_child

        # Return archive (return value is ignored by evaluator)
        return archive_X
# EVOLVE-BLOCK-END
