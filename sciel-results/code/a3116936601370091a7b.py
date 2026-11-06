"""Thompson Sampling with event-driven adaptive forgetting for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling with
    event-driven adaptive forgetting.

    Structural changes from seed:
    1. Maintains covariance matrix (Sigma) directly instead of precision (A),
       updated via matrix inversion lemma for efficiency.
    2. Event-driven adaptive forgetting: monitors Mahalanobis distance of reward
       residuals. When prediction error significantly exceeds uncertainty
       (regime shift), applies strong decay (0.5) to rapidly discard stale model.
       Otherwise, applies mild decay (0.99) for stability.
    3. Robust Cholesky-based sampling with fallbacks.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s kill

        # Per-arm posterior: mean theta (dim), covariance Sigma (dim x dim)
        # Initialize with high uncertainty (large covariance)
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        self.Sigmas = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # Forgetting parameters
        self.decay_strong = 0.5   # Strong decay for regime shifts
        self.decay_mild = 0.99    # Mild decay for normal operation
        self.mahalanobis_threshold = 3.0  # Threshold for detecting regime shift

        # Cache Cholesky factors for Sigma
        self.cholesky = [None] * n_arms

        # RNG
        self.rng = np.random.default_rng()

    def _safe_cholesky(self, Sigma, idx):
        """Compute Cholesky of covariance with regularization fallback."""
        try:
            L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(self.d))
            self.cholesky[idx] = L
            return L
        except np.linalg.LinAlgError:
            # Add increasing regularization
            for reg in [1e-6, 1e-4, 1e-2, 0.1, 1.0]:
                try:
                    L = np.linalg.cholesky(Sigma + reg * np.eye(self.d))
                    self.cholesky[idx] = L
                    return L
                except np.linalg.LinAlgError:
                    continue
            # Last resort: identity
            self.cholesky[idx] = np.eye(self.d)
            return self.cholesky[idx]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: pick arm with highest mean prediction
            means = [self.thetas[a] @ context for a in range(self.n)]
            return int(np.argmax(means))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            L = self._safe_cholesky(self.Sigmas[a], a)

            # Sample theta ~ N(self.thetas[a], self.Sigmas[a])
            # theta_sample = theta + L z, where z ~ N(0, I)
            # Because Cov(L z) = L L^T = Sigma
            z = self.rng.normal(0, 1, self.d)
            theta_sample = self.thetas[a] + L @ z
            scores.append(theta_sample @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error and Mahalanobis distance
        pred = self.thetas[arm] @ context
        residual = reward - pred

        # Compute prediction variance: var = context^T Sigma context
        L = self._safe_cholesky(self.Sigmas[arm], arm)
        # var = ||L^T context||^2
        L_T_ctx = L.T @ context
        pred_var = np.dot(L_T_ctx, L_T_ctx)
        pred_std = np.sqrt(max(pred_var, 1e-12))

        # Mahalanobis distance (normalized residual)
        mahalanobis = abs(residual) / pred_std

        # Event-driven adaptive forgetting
        if mahalanobis > self.mahalanobis_threshold:
            # Regime shift detected: strong decay to discard stale model
            decay = self.decay_strong
        else:
            # Normal operation: mild decay
            decay = self.decay_mild

        # Update covariance with decay
        # Sigma_new = decay * Sigma_old
        # Then apply matrix inversion lemma for new observation
        self.Sigmas[arm] = decay * self.Sigmas[arm]

        # Matrix inversion lemma:
        # (Sigma + x x^T)^{-1} = Sigma^{-1} - Sigma^{-1} x x^T Sigma^{-1} / (1 + x^T Sigma^{-1} x)
        # Equivalently for covariance update:
        # Sigma_new = Sigma - Sigma x (1 + x^T Sigma x)^{-1} x^T Sigma
        # But we already decayed Sigma, so let Sigma_decayed = decay * Sigma_old
        # We want to update as if we observed (x, reward) with prior (theta_old, Sigma_decayed)
        
        # Actually, let's be more careful.
        # Posterior mean update:
        # theta_new = theta_old + Sigma_decayed x (1 + x^T Sigma_decayed x)^{-1} (reward - x^T theta_old)
        # Posterior covariance update:
        # Sigma_new = Sigma_decayed - Sigma_decayed x x^T Sigma_decayed (1 + x^T Sigma_decayed x)^{-1}

        Sigma_dec = self.Sigmas[arm]
        theta_old = self.thetas[arm]

        # Compute x^T Sigma x
        Sx = Sigma_dec @ context
        x_Sx = context @ Sx
        denom = 1.0 + x_Sx

        # Update mean
        gain = Sx / denom
        self.thetas[arm] = theta_old + gain * residual

        # Update covariance
        # Sigma_new = Sigma_dec - gain * x^T * Sigma_dec  (rank-1 update)
        # gain is a vector, so gain * x^T is outer product
        # Actually: Sigma_new = Sigma_dec - (Sx / denom) * (x^T Sigma_dec)
        # = Sigma_dec - Sx * (x^T Sigma_dec) / denom
        # Note: x^T Sigma_dec is a row vector, Sx is a column vector
        self.Sigmas[arm] = Sigma_dec - np.outer(Sx, context) / denom

        # Invalidate cached Cholesky
        self.cholesky[arm] = None
# EVOLVE-BLOCK-END
