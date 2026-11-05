"""Non-stationary linear contextual bandit with regime-switching Kalman filters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-arm state-space model with regime-switching process noise.
    Maintains two Kalman filter states per arm: stable (low noise) and volatile (high noise).
    Uses innovation magnitude to probabilistically switch regimes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()

        # Process noise for stable and volatile modes
        self.Q_stable = 0.001 * np.eye(dim)
        self.Q_volatile = 0.1 * np.eye(dim)

        # Observation noise
        self.R = 0.1  # scalar observation noise variance

        # Per-arm states: (P, theta, regime, innovation_history)
        # regime: 0=stable, 1=volatile
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.regime = [0 for _ in range(n_arms)]  # 0=stable, 1=volatile
        self.innov_hist = [np.zeros(20) for _ in range(n_arms)]  # recent innovations
        self.innov_idx = [0 for _ in range(n_arms)]

        # Transition probabilities for regime switching
        self.p_stable_to_volatile = 0.05
        self.p_volatile_to_stable = 0.3

        # Counters
        self.round = 0

    def _kalman_predict(self, arm):
        """Predict step: inflate covariance based on current regime."""
        Q = self.Q_stable if self.regime[arm] == 0 else self.Q_volatile
        self.P[arm] = self.P[arm] + Q
        return self.P[arm]

    def _kalman_update(self, arm, context, reward):
        """Update step: standard Kalman filter update."""
        x = context
        P = self.P[arm]
        theta = self.theta[arm]

        # Predicted reward
        pred = theta @ x

        # Innovation
        innov = reward - pred

        # Innovation variance
        S = x @ P @ x + self.R

        # Kalman gain
        K = (P @ x) / S

        # Update theta
        self.theta[arm] = theta + K * innov

        # Update P (Joseph form for numerical stability)
        I_Kx = np.eye(self.d) - np.outer(K, x)
        self.P[arm] = I_Kx @ P @ I_Kx.T + self.R * np.outer(K, K)

        # Store innovation for regime detection
        self.innov_hist[arm][self.innov_idx[arm]] = abs(innov)
        self.innov_idx[arm] = (self.innov_idx[arm] + 1) % 20

        return innov, S

    def _update_regime(self, arm):
        """Update regime based on recent innovation magnitudes."""
        hist = self.innov_hist[arm]
        mean_innov = np.mean(hist)

        # If innovations are large, shift toward volatile; if small, shift toward stable
        if mean_innov > 0.5:
            # Increase probability of volatile
            if self.regime[arm] == 0:
                if np.random.random() < self.p_stable_to_volatile * (1 + mean_innov):
                    self.regime[arm] = 1
            else:
                # Stay volatile with high probability
                if np.random.random() < self.p_volatile_to_stable:
                    self.regime[arm] = 0
        else:
            # Shift toward stable
            if self.regime[arm] == 1:
                if np.random.random() < self.p_volatile_to_stable:
                    self.regime[arm] = 0
            else:
                # Stay stable
                pass

    def select(self, context):
        # Deadline guard
        if time.time() - self.t0 > 110:
            return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))

        scores = []
        for a in range(self.n):
            # Predict step
            P = self._kalman_predict(a)
            theta = self.theta[a]

            # Mean prediction
            mean = theta @ context

            # Uncertainty
            var = context @ P @ context + self.R
            std = np.sqrt(max(var, 1e-8))

            # UCB-style score with adaptive exploration
            # More exploration in volatile regime
            alpha = 0.5 if self.regime[a] == 0 else 1.0
            score = mean + alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round += 1

        # Update step
        innov, S = self._kalman_update(arm, context, reward)

        # Regime update
        self._update_regime(arm)

        # For other arms, just do prediction (no observation)
        for a in range(self.n):
            if a != arm:
                self._kalman_predict(a)
# EVOLVE-BLOCK-END
