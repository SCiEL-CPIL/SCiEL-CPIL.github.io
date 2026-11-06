"""
Bayesian Linear Bandit with event-driven adaptive forgetting.
Structural change: single model per arm with regime-shift detection.
When prediction error exceeds uncertainty (normalized residual), apply strong decay;
otherwise mild decay. This directly targets non-stationarity.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t_start = time.time()
        self.deadline = self.t_start + 115.0

        # Bayesian linear model per arm
        # A[a] is d x d, b[a] is d-vector
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Cholesky factors for each arm
        self.L = [np.linalg.cholesky(self.A[a]) for a in range(n_arms)]

        # Forgetting parameters
        self.mild_lambda = 0.99      # mild decay when no regime shift
        self.strong_lambda = 0.50    # strong decay on regime shift
        self.regime_threshold = 2.0  # normalized residual threshold for regime shift

        # Track count of observations per arm for initialization
        self.counts = [0 for _ in range(n_arms)]

    def _update_chol(self, A):
        """Update Cholesky factor with fallback regularization."""
        try:
            return np.linalg.cholesky(A + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            A = A + 1e-4 * np.eye(self.d)
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                return np.eye(self.d)

    def _get_theta(self, arm):
        """Get point estimate theta = A^{-1} b."""
        try:
            L = self.L[arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            return theta
        except Exception:
            return np.zeros(self.d)

    def _get_theta_and_cov(self, arm):
        """Get theta and posterior covariance (as Cholesky)."""
        try:
            L = self.L[arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            return theta, L
        except Exception:
            return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self._get_theta(a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            theta, L = self._get_theta_and_cov(a)

            try:
                # Thompson Sampling: sample from posterior
                z = np.random.randn(self.d)
                # Posterior covariance is A^{-1} = L^{-1} L^{-T}
                # Sample: theta_sample = theta + A^{-1/2} z = theta + L^{-T} z
                noise = np.linalg.solve(L.T, z)
                theta_sample = theta + noise
                score = theta_sample @ context
            except Exception:
                score = theta @ context

            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute current prediction and uncertainty
        theta = self._get_theta(arm)
        pred = theta @ context

        # Compute prediction variance (diagonal of posterior covariance for this context)
        # Var(y|context) = context^T A^{-1} context
        try:
            L = self.L[arm]
            # A^{-1} context = L^{-1} L^{-T} context
            temp = np.linalg.solve(L.T, context)
            var = np.dot(context, np.linalg.solve(L, temp))
        except Exception:
            var = 1.0

        # Normalized residual (z-score)
        if var > 1e-12:
            normalized_resid = (reward - pred) / np.sqrt(var)
        else:
            normalized_resid = 0.0

        # Determine forgetting factor based on regime shift detection
        if abs(normalized_resid) > self.regime_threshold:
            lam = self.strong_lambda
        else:
            lam = self.mild_lambda

        # Update A and b with forgetting
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        self.L[arm] = self._update_chol(self.A[arm])

        self.counts[arm] += 1
# EVOLVE-BLOCK-END
