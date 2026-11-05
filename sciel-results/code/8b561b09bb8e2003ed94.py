"""Thompson Sampling with Kalman filter + adaptive observation noise (R) and process noise (Q)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Thompson Sampling: sample theta from posterior N(theta_hat, P) for each arm
    - Kalman filter with Joseph-form covariance update
    - Adaptive observation noise R (EMA of squared residuals)
    - Adaptive process noise Q (EMA of residual magnitude)
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 125.0  # Safety margin before 150s kill
        self._rng = np.random.default_rng(42)

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive observation noise R (per-arm)
        self.R = [1.0 for _ in range(n_arms)]
        self.residual_sq_ema = [1.0 for _ in range(n_arms)]  # EMA of y^2
        self.r_ema_alpha = 0.1  # Smoothing for R

        # Adaptive process noise Q (per-arm)
        self.Q_scale = [0.01 for _ in range(n_arms)]
        self.residual_abs_ema = [0.0 for _ in range(n_arms)]  # EMA of |y|
        self.q_ema_alpha = 0.1  # Smoothing for Q

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_Q(self, arm):
        """Get process noise matrix from scalar scale."""
        return self.Q_scale[arm] * np.eye(self.d)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if time.time() - self._t0 > self._deadline:
            scores = []
            for a in range(self.n):
                mu = self.theta[a] @ context
                scores.append(mu)
            return int(np.argmax(scores))

        # Thompson Sampling: sample theta from posterior for each arm
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            P_pred = self.P[a] + self._get_Q(a)

            # Sample theta from N(theta[a], P_pred)
            # Use Cholesky for efficient sampling
            try:
                L = np.linalg.cholesky(P_pred)
                z = self._rng.standard_normal(self.d)
                theta_sample = self.theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(P_pred)
                eigvals = np.maximum(eigvals, 0)
                z = self._rng.standard_normal(self.d)
                theta_sample = self.theta[a] + eigvecs @ (np.sqrt(eigvals) * z)

            # Score = sampled theta dot context
            score = theta_sample @ context
            scores.append(score)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        a = arm
        self.t += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # Predict step: inflate covariance with process noise
        Q_a = self._get_Q(a)
        P_pred = self.P[a] + Q_a

        # Innovation (residual)
        y = reward - self.theta[a] @ context
        R_a = self.R[a]
        S = context @ P_pred @ context + R_a

        # Kalman gain
        K = P_pred @ context / max(S, 1e-12)

        # Update state
        self.theta[a] = self.theta[a] + K * y

        # Joseph form: P = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        IK = np.eye(self.d) - np.outer(K, context)
        self.P[a] = IK @ P_pred @ IK.T + np.outer(K, K) * R_a

        # Ensure symmetry and PSD
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        self.P[a] += 1e-8 * np.eye(self.d)

        self.count[a] += 1

        # Adaptive observation noise R: EMA of squared residuals
        y_sq = y * y
        self.residual_sq_ema[a] = (1 - self.r_ema_alpha) * self.residual_sq_ema[a] + self.r_ema_alpha * y_sq
        # Bounded: between 0.1 and 10.0
        self.R[a] = np.clip(self.residual_sq_ema[a], 0.1, 10.0)

        # Adaptive process noise Q: EMA of |residual|
        abs_res = abs(y)
        self.residual_abs_ema[a] = (1 - self.q_ema_alpha) * self.residual_abs_ema[a] + self.q_ema_alpha * abs_res
        # Map to Q scale with tanh saturation
        raw_scale = self.residual_abs_ema[a] * 0.15
        q_scale = 0.001 + 0.499 * np.tanh(raw_scale)
        self.Q_scale[a] = q_scale
# EVOLVE-BLOCK-END
