# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """k-NN surrogate + epsilon-greedy (max-min distance / kNN predicted value)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s wall-clock kill

        # Storage for samples
        X = np.empty((budget, dim))
        F = np.empty(budget)
        n = 0

        # Initialize with a small Latin-hypercube-like spread for coverage
        # Use a few deterministic + random points to seed the surrogate
        init_n = min(8, budget)
        for i in range(init_n):
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            else:
                # spread out corners
                x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            X[i] = x
            F[i] = func(x)
            n = i + 1

        best_x = X[np.argmin(F[:n])]
        best_f = F[:n].min()

        k = 3  # number of neighbors for kNN prediction

        for step in range(n, budget):
            if time.time() >= deadline:
                break

            # --- Build kNN surrogate prediction for candidate selection ---
            # We need to choose the next point. Use epsilon-greedy:
            #   - exploitation: minimize predicted f (kNN mean of k nearest)
            #   - exploration: maximize distance to nearest existing sample
            # We generate a set of candidates and pick the best by a combined score.

            # Generate candidates:
            # 1) Random points
            # 2) Points near current best (local refinement)
            # 3) Far points (exploration)
            n_cand = 200

            candidates = np.empty((n_cand, dim))

            # 60% random
            n_rand = int(n_cand * 0.6)
            candidates[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))

            # 25% near best
            n_near = int(n_cand * 0.25)
            local_sigma = 0.5
            candidates[n_rand:n_rand + n_near] = np.clip(
                best_x[None, :] + local_sigma * rng.standard_normal((n_near, dim)), -5, 5
            )

            # 15% far from all existing (max-min distance heuristic)
            n_far = n_cand - n_rand - n_near
            if n > 0:
                # Sample many and pick those with high min-distance
                far_pool = rng.uniform(-5, 5, (n_far * 5, dim))
                # Compute min distance to existing samples
                # For efficiency, batch it
                dists = np.min(
                    np.linalg.norm(far_pool[:, None, :] - X[:n][None, :, :], axis=2),
                    axis=1
                )
                far_idx = np.argsort(dists)[-n_far:]
                candidates[n_rand + n_near:] = far_pool[far_idx]
            else:
                candidates[n_rand + n_near:] = rng.uniform(-5, 5, (n_far, dim))

            # --- Score each candidate ---
            # kNN predicted value: mean of k nearest neighbors' F
            # Distance to nearest existing sample: for exploration

            # Compute distances from candidates to existing samples
            # candidates: (n_cand, dim), X[:n]: (n, dim)
            # For memory, do it in chunks if needed, but n_cand=200, n<=500, dim=5 -> fine
            d = np.linalg.norm(candidates[:, None, :] - X[:n][None, :, :], axis=2)  # (n_cand, n)

            # kNN prediction: mean of k smallest distances' F
            k_eff = min(k, n)
            # Get indices of k nearest
            k_idx = np.argpartition(d, k_eff, axis=1)[:, :k_eff]  # (n_cand, k_eff)
            k_vals = np.take_along_axis(F[:n][None, :], k_idx, axis=1)  # (n_cand, k_eff)
            pred_f = k_vals.mean(axis=1)  # (n_cand,)

            # Min distance to any existing sample (for exploration)
            min_dist = d.min(axis=1)  # (n_cand,)

            # Normalize for combination
            pred_min, pred_max = pred_f.min(), pred_f.max()
            pred_range = pred_max - pred_min + 1e-12
            pred_norm = (pred_f - pred_min) / pred_range  # lower is better

            dist_min, dist_max = min_dist.min(), min_dist.max()
            dist_range = dist_max - dist_min + 1e-12
            dist_norm = (min_dist - dist_min) / dist_range  # higher is better (more exploration)

            # Epsilon-greedy: with probability epsilon, pick max exploration; else pick min prediction
            # We'll use a combined score: score = pred_norm - lambda * dist_norm
            # Lower combined score is better (low prediction, high distance)
            # Actually: we want to MINIMIZE pred and MAXIMIZE dist.
            # Combined: cost = pred_norm - dist_norm  (minimize this)
            # But early on we want more exploration. Adapt lambda over time.

            progress = step / max(budget - 1, 1)
            # Early: more exploration (high lambda), Late: more exploitation (low lambda)
            lambda_expl = 1.0 * (1.0 - progress) + 0.1

            cost = pred_norm - lambda_expl * dist_norm
            best_cand_idx = np.argmin(cost)

            x_new = candidates[best_cand_idx]
            x_new = np.clip(x_new, -5, 5)

            f_new = func(x_new)
            X[step] = x_new
            F[step] = f_new
            n = step + 1

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
