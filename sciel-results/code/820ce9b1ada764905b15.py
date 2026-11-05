"""Rewritten program: Non-stationary linear contextual bandit with dual-timescale tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Dual-timescale ridge regression (fast + slow) for robust tracking
    - Thompson Sampling for exploration
    - Adaptive forgetting based on prediction error
    - Efficient Cholesky-based sampling
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t_start = time.time()
        self.deadline = self.t_start + 115.0  # safety margin before 150s limit

        # Primary model (fast, low regularization)
        self.A_fast = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_fast = [np.zeros(dim) for _ in range(n_arms)]

        # Secondary model (slow, high regularization) - acts as anchor
        self.A_slow = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_slow = [np.zeros(dim) for _ in range(n_arms)]

        # Decay parameters
        self.decay_fast = 0.97
        self.decay_slow = 0.995
        self.min_decay_fast = 0.85
        self.max_decay_fast = 0.995
        self.adaptation_rate = 0.15

        # Residual tracking for adaptive decay
        self.residuals = [[] for _ in range(n_arms)]
        self.residual_window = 30

        # Precompute Cholesky factors
        self._L_fast = [np.linalg.cholesky(self.A_fast[a]) for a in range(n_arms)]
        self._L_slow = [np.linalg.cholesky(self.A_slow[a]) for a in range(n_arms)]

        # Track which arm is currently "hot" (recently had low residuals)
        self.arm_confidence = [1.0] * n_arms

    def _update_chol(self, arm, A, L):
        """Update Cholesky factor, with fallback regularization."""
        try:
            return np.linalg.cholesky(A + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            A = A + 1e-4 * np.eye(self.d)
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                return np.eye(self.d)

    def _get_theta_cov(self, arm, use_fast=True):
        """Get theta and covariance (via Cholesky) for sampling."""
        if use_fast:
            A = self.A_fast[arm]
            b = self.b_fast[arm]
            L = self._L_fast[arm]
        else:
            A = self.A_slow[arm]
            b = self.b_slow[arm]
            L = self._L_slow[arm]
        
        try:
            theta = np.linalg.solve(L.T, np.linalg.solve(L, b))
            return theta, L
        except Exception:
            return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta_f, _ = self._get_theta_cov(a, True)
                    theta_s, _ = self._get_theta_cov(a, False)
                    score = 0.7 * (theta_f @ context) + 0.3 * (theta_s @ context)
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    pass
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            try:
                # Blend fast and slow estimates for Thompson Sampling
                theta_f, L_f = self._get_theta_cov(a, True)
                theta_s, L_s = self._get_theta_cov(a, False)

                # Blend: 70% fast, 30% slow
                theta_blend = 0.7 * theta_f + 0.3 * theta_s

                # For uncertainty, use the fast model's covariance (more responsive)
                # Sample: theta ~ N(theta_blend, A_fast^{-1})
                z = np.random.randn(self.d)
                # A^{-1} = L^{-T} L^{-1}, so sqrt(A^{-1}) z = L^{-T} z
                noise = np.linalg.solve(L_f.T, z)
                theta_sample = theta_blend + noise

                scores.append(theta_sample @ context)
            except Exception:
                # Fallback: use point estimate
                try:
                    theta_f, _ = self._get_theta_cov(a, True)
                    scores.append(theta_f @ context)
                except:
                    scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error for adaptive decay
        try:
            theta_f, _ = self._get_theta_cov(arm, True)
            pred = theta_f @ context
            residual = reward - pred
            # Normalize by uncertainty
            _, L_f = self._get_theta_cov(arm, True)
            ctx_var = context @ np.linalg.solve(self.A_fast[arm], context)
            mahalanobis = abs(residual) / np.sqrt(ctx_var + 1e-8)
        except Exception:
            mahalanobis = 1.0

        # Update residual history
        self.residuals[arm].append(mahalanobis)
        if len(self.residuals[arm]) > self.residual_window:
            self.residuals[arm].pop(0)

        # Adaptive decay: large residuals -> faster forgetting
        if mahalanobis > 2.5:
            target_decay = self.min_decay_fast
        elif mahalanobis > 1.5:
            target_decay = 0.93
        else:
            mean_resid = np.mean(self.residuals[arm]) if self.residuals[arm] else 0
            if mean_resid > 1.5:
                target_decay = 0.95
            else:
                target_decay = self.decay_fast

        self.decay_fast += self.adaptation_rate * (target_decay - self.decay_fast)
        self.decay_fast = np.clip(self.decay_fast, self.min_decay_fast, self.max_decay_fast)

        decay_f = self.decay_fast
        decay_s = self.decay_slow

        # Update fast model
        self.A_fast[arm] = decay_f * self.A_fast[arm] + np.outer(context, context)
        self.b_fast[arm] = decay_f * self.b_fast[arm] + reward * context
        self._L_fast[arm] = self._update_chol(arm, self.A_fast[arm], self._L_fast[arm])

        # Update slow model
        self.A_slow[arm] = decay_s * self.A_slow[arm] + np.outer(context, context)
        self.b_slow[arm] = decay_s * self.b_slow[arm] + reward * context
        self._L_slow[arm] = self._update_chol(arm, self.A_slow[arm], self._L_slow[arm])

        # Update arm confidence
        if mahalanobis < 1.0:
            self.arm_confidence[arm] = min(1.0, self.arm_confidence[arm] + 0.05)
        else:
            self.arm_confidence[arm] = max(0.0, self.arm_confidence[arm] - 0.05)
# EVOLVE-BLOCK-END
