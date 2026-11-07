"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Gaussian Process-based multi-objective Bayesian optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        archive_X = []
        archive_F = []
        eval_count = 0

        # GP hyperparameters
        length_scale = 0.3
        signal_var = 1.0
        noise_var = 1e-6

        def _kern(A, B):
            """Squared exponential kernel between two sets of points."""
            # A: (n, d), B: (m, d)
            d2 = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            d2 = np.maximum(d2, 0)
            return signal_var * np.exp(-0.5 * d2 / (length_scale ** 2))

        def _gp_predict(X_train, Y_train, X_test):
            """GP prediction: mean and std at X_test given training data."""
            n = len(X_train)
            if n == 0:
                return np.zeros(len(X_test)), np.sqrt(signal_var + noise_var) * np.ones(len(X_test))
            K = _kern(X_train, X_train) + noise_var * np.eye(n)
            Kf = _kern(X_train, X_test)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                K_reg = K + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K_reg)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mean = Kf.T @ alpha
            v = np.linalg.solve(L, Kf)
            var = signal_var - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)
            return mean, np.sqrt(var)

        def is_dominated(f, archive_F_arr):
            """Check if f is dominated by any point in archive_F_arr."""
            if len(archive_F_arr) == 0:
                return False
            leq = np.all(archive_F_arr <= f, axis=1)
            lt = np.any(archive_F_arr < f, axis=1)
            return np.any(leq & lt)

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            if archive_F and is_dominated(f, np.array(archive_F)):
                return
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            eval_count += 1
            add_to_archive(x_clipped.copy(), f)
            return f

        # Phase 1: Initial sampling - mix of random and targeted
        n_init = min(25, budget)
        for i in range(n_init):
            if eval_count >= budget:
                break
            if i < 5:
                # Targeted: push x_2..x_n toward 0
                x = np.zeros(dim)
                x[0] = rng.random()
                for j in range(1, dim):
                    x[j] = rng.random() * 0.05
            elif i < 15:
                # Random
                x = lo + (hi - lo) * rng.random(dim)
            else:
                # Biased toward low x_2..x_n
                x = lo + (hi - lo) * rng.random(dim)
                x[1:] *= 0.1
            evaluate(x)

        # Phase 2: GP-based Bayesian optimization
        # We'll use the GP to model f2 (the penalty-dominated objective)
        # and use an acquisition function to find points with low f2
        # while also maintaining diversity in f1

        # Build training data from archive
        def build_training_data():
            if not archive_X:
                return None, None
            X = np.array(archive_X)
            F = np.array(archive_F)
            return X, F

        # Acquisition: expected improvement on f2, weighted by novelty in f1
        def acquisition_score(X_candidates, X_train, Y_train_f2, archive_F):
            """Score candidates by expected improvement on f2 + diversity bonus."""
            n_cand = len(X_candidates)
            scores = np.zeros(n_cand)
            if len(X_train) == 0:
                return scores

            # Predict f2 at candidates
            mean, std = _gp_predict(X_train, Y_train_f2, X_candidates)

            # Current best f2
            best_f2 = np.min(Y_train_f2) if len(Y_train_f2) > 0 else 11.0

            # Expected improvement
            delta = best_f2 - mean
            z = delta / (std + 1e-10)
            EI = delta * norm.cdf(z) + std * norm.pdf(z)
            EI = np.maximum(EI, 0)

            # Diversity bonus: distance from archive in f-space
            if len(archive_F) > 0:
                archive_F_arr = np.array(archive_F)
                # For each candidate, compute min distance to archive in f-space
                # We need to predict f1 too, or use a heuristic
                # Use the candidate's predicted f2 and a heuristic for f1
                # Simpler: use input-space distance
                min_dists = np.min(np.linalg.norm(X_candidates[:, None, :] - X_train[None, :, :], axis=2), axis=1)
                div_bonus = 0.3 * min_dists
            else:
                div_bonus = np.zeros(n_cand)

            scores = EI + div_bonus
            return scores

        # Phase 2a: GP-guided search for low f2
        n_gp_iter = min(50, budget - eval_count)
        for it in range(n_gp_iter):
            if eval_count >= budget:
                break

            X_train, F_train = build_training_data()
            if X_train is None or len(X_train) < 3:
                # Not enough data, do random
                x = lo + (hi - lo) * rng.random(dim)
                x[1:] *= 0.1  # bias toward low penalty
                evaluate(x)
                continue

            Y_train_f2 = F_train[:, 1]

            # Generate candidate points
            n_cand = 200
            X_cand = lo + (hi - lo) * rng.random((n_cand, dim))
            # Add some targeted candidates
            for k in range(20):
                idx = rng.integers(0, n_cand)
                X_cand[idx, 1:] *= 0.05  # push penalty vars toward 0

            # Score candidates
            scores = acquisition_score(X_cand, X_train, Y_train_f2, archive_F)
            best_idx = np.argmax(scores)
            x_new = X_cand[best_idx]
            evaluate(x_new)

        # Phase 3: Front exploration - sample along x_1 with x_2..x_n near 0
        n_front = min(40, budget - eval_count)
        for i in range(n_front):
            if eval_count >= budget:
                break
            x1 = i / max(n_front - 1, 1)
            x = np.zeros(dim)
            x[0] = x1
            # Small noise in other dimensions
            x[1:] = rng.random(dim - 1) * 0.02
            evaluate(x)

        # Phase 4: Local refinement around best archive points
        remaining = budget - eval_count
        if remaining > 0 and archive_X:
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            # Sort by f2
            idx_sorted = np.argsort(archive_F_arr[:, 1])[:min(8, len(archive_F))]

            for idx in idx_sorted:
                if eval_count >= budget:
                    break
                x_center = archive_X_arr[idx].copy()
                n_local = min(3, budget - eval_count)
                for _ in range(n_local):
                    if eval_count >= budget:
                        break
                    # Perturb, keeping x_2..x_n small
                    x_new = x_center.copy()
                    x_new[0] += 0.05 * rng.standard_normal()
                    x_new[1:] += 0.02 * rng.standard_normal(dim - 1)
                    x_new = np.clip(x_new, lo, hi)
                    evaluate(x_new)

        return archive_X if archive_X else [lo]
# EVOLVE-BLOCK-END
