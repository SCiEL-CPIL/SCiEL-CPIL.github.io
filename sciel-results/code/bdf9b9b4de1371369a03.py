"""Thompson Sampling with adaptive sliding-window posterior for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling with
    adaptive recency weighting and robust Cholesky-based sampling.
    
    Strategy:
    1. Maintain a recency-weighted (exponential forgetting) posterior for each arm.
    2. Use Thompson Sampling: draw theta ~ N(mean, covariance) and pick argmax theta @ x.
    3. Adaptive forgetting: increase forgetting rate when prediction error is large
       (Mahalanobis distance), allowing faster recovery from regime shifts.
    4. Robust Cholesky fallback to prevent crashes on near-singular matrices.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 110  # safety margin before 120s kill

        # Per-arm posterior: A (dim x dim), b (dim)
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor (base)
        self.base_alpha = 0.98
        # Adaptive forgetting: will adjust per-update based on surprise
        self.alpha = self.base_alpha

        # Track prediction errors for adaptive forgetting
        self.pred_errors = [0.0] * n_arms

        # Cache Cholesky factors
        self.cholesky = [None] * n_arms

    def _safe_cholesky(self, A, idx):
        """Compute Cholesky with regularization fallback."""
        try:
            L = np.linalg.cholesky(A)
            self.cholesky[idx] = L
            return L
        except np.linalg.LinAlgError:
            # Add small regularization and retry
            for reg in [1e-6, 1e-4, 1e-2, 1.0]:
                try:
                    L = np.linalg.cholesky(A + reg * np.eye(self.d))
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
            means = [np.linalg.solve(self.A[a], self.b[a]) @ context for a in range(self.n)]
            return int(np.argmax(means))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            L = self._safe_cholesky(self.A[a], a)
            # Mean: solve A theta = b
            try:
                theta_mean = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta_mean = np.zeros(self.d)

            # Sample theta ~ N(theta_mean, A^{-1})
            # A^{-1} = L^{-1} L^{-T}, so sample = theta_mean + L^{-T} z where z ~ N(0, I)
            # Actually: A = L L^T, so A^{-1} = L^{-T} L^{-1}
            # Covariance = A^{-1}. Sample: theta = theta_mean + A^{-1/2} z
            # A^{-1/2} = L^{-T} (since A^{1/2} = L, so A^{-1/2} = L^{-T})
            # Wait: A = L L^T, so A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
            # A^{-1/2} such that (A^{-1/2})^T A^{-1/2} = A^{-1}
            # Let A^{-1/2} = L^{-T}, then (L^{-T})^T L^{-T} = L^{-1} L^{-T} ... not quite.
            # Better: sample z ~ N(0, I), then theta = theta_mean + L^{-T} z
            # Because Cov(L^{-T} z) = L^{-T} L^{-1} = A^{-1}. Yes!
            z = np.random.randn(self.d)
            # Solve L w = z, then w = L^{-1} z; we need L^{-T} z
            # L^{-T} z = (L^{-1} z)^T ... no. Let's do: solve L^T v = z => v = L^{-T} z
            try:
                v = np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                v = z

            theta_sample = theta_mean + v
            scores.append(theta_sample @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error (Mahalanobis-like surprise)
        try:
            theta_mean = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta_mean @ context
            # Variance of prediction
            L = self._safe_cholesky(self.A[arm], arm)
            var = context @ np.linalg.solve(L, np.linalg.solve(L.T, context))
            std = np.sqrt(max(var, 1e-12))
            surprise = abs(reward - pred) / std
        except Exception:
            surprise = 1.0

        # Adaptive forgetting: larger surprise -> faster forgetting (smaller alpha)
        # alpha in (0, 1); smaller alpha means faster forgetting
        # Map surprise to alpha: high surprise -> alpha closer to 0.9, low -> 0.98
        if surprise > 3.0:
            self.alpha = 0.90
        elif surprise > 2.0:
            self.alpha = 0.94
        elif surprise > 1.0:
            self.alpha = 0.97
        else:
            self.alpha = 0.98

        # Update with forgetting
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context

        # Invalidate cached Cholesky
        self.cholesky[arm] = None
# EVOLVE-BLOCK-END
