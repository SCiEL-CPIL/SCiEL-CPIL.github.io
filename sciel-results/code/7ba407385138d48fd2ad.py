"""Surrogate-based optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-based optimizer using RBF with Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling (50 evaluations)
        n_init = 50
        X = np.zeros((n_init, dim))
        F = np.zeros(n_init)

        # Latin Hypercube Sampling for initial points
        for i in range(n_init):
            X[i] = rng.uniform(-5, 5, dim)
            F[i] = func(X[i])

        # Phase 2: Surrogate-based optimization
        # Use RBF surrogate with Expected Improvement acquisition

        # Parameters for RBF
        # We'll use a simple RBF: f(x) = sum_i alpha_i * exp(-beta * ||x - x_i||^2)
        # But fitting this exactly is expensive. Let's use a simpler approach.

        # Alternative: Use a weighted average of nearby points (local surrogate)
        # Or use scipy's RBF interpolator

        # Let's use a simpler approach: 
        # 1. Find the current best point
        # 2. Generate candidate points around it
        # 3. Use a simple heuristic to select the next point

        # Actually, let's implement a proper RBF-based surrogate
        # with Expected Improvement

        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]

        # RBF parameters
        # Use a fixed length scale
        length_scale = 1.0
        beta = 1.0 / (2.0 * length_scale**2)

        n_remaining = budget - n_init
        for step in range(n_remaining):
            # Check time
            if time.time() > deadline:
                break

            # Build RBF surrogate
            # f(x) = sum_i alpha_i * exp(-beta * ||x - x_i||^2)
            # Fit alpha by solving linear system
            K = np.zeros((n_init, n_init))
            for i in range(n_init):
                for j in range(n_init):
                    d2 = np.sum((X[i] - X[j])**2)
                    K[i, j] = np.exp(-beta * d2)

            # Add small regularization for numerical stability
            K_reg = K + 1e-6 * np.eye(n_init)

            # Solve for alpha
            try:
                alpha = np.linalg.solve(K_reg, F)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_reg, F, rcond=None)[0]

            # Define surrogate function
            def surrogate(x):
                d2 = np.sum((x - X)**2, axis=1)
                return np.sum(alpha * np.exp(-beta * d2))

            # Expected Improvement acquisition
            # EI(x) = (mu(x) - f_best) * Phi((mu(x) - f_best)/sigma(x)) + sigma(x) * phi((mu(x) - f_best)/sigma(x))
            # For deterministic RBF, we need to add some uncertainty estimate
            # Let's use a simple heuristic: uncertainty decreases with density of nearby points

            # Generate candidate points
            n_candidates = 50
            candidates = np.zeros((n_candidates, dim))

            # Mix of local and global exploration
            # 70% local around best, 30% global
            n_local = int(0.7 * n_candidates)
            n_global = n_candidates - n_local

            # Local candidates around best point
            for i in range(n_local):
                step_size = 0.5 + 0.5 * (step / max(1, n_remaining))
                candidates[i] = best_x + step_size * rng.standard_normal(dim)
                candidates[i] = np.clip(candidates[i], -5, 5)

            # Global candidates
            for i in range(n_local, n_candidates):
                candidates[i] = rng.uniform(-5, 5, dim)

            # Evaluate acquisition function for each candidate
            # Use a simple EI-like score
            f_best_current = np.min(F)
            scores = np.zeros(n_candidates)

            for i in range(n_candidates):
                x = candidates[i]
                # Surrogate prediction
                d2 = np.sum((x - X)**2, axis=1)
                mu = np.sum(alpha * np.exp(-beta * d2))

                # Uncertainty estimate: based on distance to nearest points
                min_dist = np.min(np.sqrt(d2))
                sigma = 0.1 + 0.5 * min_dist  # heuristic uncertainty

                # EI calculation
                improvement = f_best_current - mu
                if sigma > 1e-10:
                    z = improvement / sigma
                    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    ei = max(0, improvement)

                scores[i] = ei

            # Select the best candidate
            best_candidate_idx = np.argmax(scores)
            x_new = candidates[best_candidate_idx].copy()
            f_new = func(x_new)

            # Add to dataset
            X = np.vstack([X, x_new.reshape(1, -1)])
            F = np.append(F, f_new)
            n_init += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
