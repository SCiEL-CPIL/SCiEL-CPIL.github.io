# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandits.

    Models each arm's weight vector as a random walk with adaptive process noise.
    Uses a fast timescale for reward estimation and a slow timescale for drift tracking.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._deadline = time.time() + 140.0  # safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]  # covariance, start moderately large
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise per arm
        self.process_noise = [0.01 for _ in range(n_arms)]
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history = 50

        # Forgetting factor for fast adaptation
        self.forget = 0.99
        # Process noise growth rate
        self.noise_growth = 1.005

    def select(self, context):
        # Deadline guard: if we're running out of time, return a simple heuristic
        if time.time() > self._deadline:
            # Return arm with highest estimated theta @ context
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Prediction: theta unchanged, P increases by process noise
            P_pred = self.P[a] + self.process_noise[a] * np.eye(self.d)
            theta_pred = self.theta[a]

            # UCB score
            inv = np.linalg.inv(P_pred)
            mean = theta_pred @ context
            var = context @ inv @ context
            # Adaptive exploration: scale with sqrt of variance
            alpha = 1.0
            score = mean + alpha * np.sqrt(var)
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Prediction step: increase covariance by process noise
        self.P[arm] = self.P[arm] + self.process_noise[arm] * np.eye(self.d)

        # Update step: Kalman filter update
        P = self.P[arm]
        H = context
        r = reward - self.theta[arm] @ context  # innovation

        S = H @ P @ H + 1.0  # innovation variance
        K = (P @ H) / S  # Kalman gain (scalar denominator)

        self.theta[arm] = self.theta[arm] + K * r
        self.P[arm] = P - np.outer(K, H @ P)

        # Ensure P stays positive definite
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

        self.count[arm] += 1

        # Track reward history for adaptive process noise
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history:
            self.reward_history[arm].pop(0)

        # Adapt process noise based on recent reward variance
        if len(self.reward_history[arm]) >= 10:
            recent = np.array(self.reward_history[arm])
            reward_var = np.var(recent)
            # If rewards are more variable, increase process noise
            target_noise = max(0.001, min(0.1, reward_var * 0.1))
            # Smoothly adjust process noise
            self.process_noise[arm] = 0.95 * self.process_noise[arm] + 0.05 * target_noise
# EVOLVE-BLOCK-END
