# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Combines the per-arm residual-driven Q adaptation from the current program
    with the Joseph-form covariance update and dual-timescale error tracking
    from the inspirations.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 130.0  # safety margin before 150s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]  # covariance
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (per-arm)
        self.Q = [0.02 * np.eye(dim) for _ in range(n_arms)]
        self.residual_hist = [[] for _ in range(n_arms)]

        # Exploration constant
        self.alpha = 1.2

        # For fast fallback if computation is too slow
        self._use_fast = False

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            # Return best arm based on current theta estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            P_pred = self.P[a] + self.Q[a]

            # Compute mean and variance of reward estimate
            mean = self.theta[a] @ context
            var = context @ P_pred @ context
            ucb = mean + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        a = arm
        self.t += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # Predict step
        P_pred = self.P[a] + self.Q[a]

        # Innovation (residual)
        y = reward - self.theta[a] @ context
        S = context @ P_pred @ context + 1.0  # observation noise = 1

        # Kalman gain
        K = P_pred @ context / max(S, 1e-12)

        # Update state using Joseph form for numerical stability
        self.theta[a] = self.theta[a] + K * y

        # Joseph form: P = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        IK = np.eye(self.d) - np.outer(K, context)
        self.P[a] = IK @ P_pred @ IK.T + np.outer(K, K) * 1.0

        # Ensure symmetry
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        # Clip small negative eigenvalues via eigendecomposition (d is small)
        if self.d <= 10:
            eigvals, eigvecs = np.linalg.eigh(self.P[a])
            eigvals = np.maximum(eigvals, 1e-10)
            self.P[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            # Fallback: add small regularization to ensure PSD
            self.P[a] += 1e-10 * np.eye(self.d)

        self.count[a] += 1

        # Adaptive process noise: track residual magnitude
        self.residual_hist[a].append(abs(y))
        if len(self.residual_hist[a]) > 20:
            self.residual_hist[a].pop(0)

        # Update Q based on residual statistics with smooth adaptation
        if self.count[a] >= 5:
            mean_res = np.mean(self.residual_hist[a])
            std_res = np.std(self.residual_hist[a]) if len(self.residual_hist[a]) > 2 else 0.0
            # Combine mean and std for robust drift estimation
            drift_est = 0.7 * mean_res + 0.3 * std_res
            # Smooth Q update: blend with previous Q to avoid abrupt changes
            q_scale = max(0.002, min(0.5, drift_est * 0.15))
            old_q = self.Q[a][0, 0]
            new_q = 0.8 * old_q + 0.2 * q_scale
            self.Q[a] = new_q * np.eye(self.d)
# EVOLVE-BLOCK-END
