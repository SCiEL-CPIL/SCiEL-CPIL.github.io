"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Regime-switching Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Each arm maintains a single theta but two covariance matrices representing
    'stable' (low process noise) and 'volatile' (high process noise) regimes.
    A per-arm regime probability is updated via a smooth sigmoid of the
    normalized innovation, blending the two covariances for both UCB scoring
    and Kalman updates.  This explicitly distinguishes gradual drift from
    abrupt structural breaks.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Two covariance matrices per arm: stable and volatile
        self.P_stable = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.P_volatile = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Process noise levels for each regime
        self._Q_stable = 5e-4
        self._Q_volatile = 0.03

        # Regime probability per arm (0 = stable, 1 = volatile)
        self._regime_prob = [0.0 for _ in range(n_arms)]
        # EMA of squared normalized innovations for regime detection
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.2

        # Sigmoid parameters for regime switching
        self._sigmoid_center = 1.0   # innovation norm threshold
        self._sigmoid_steepness = 3.0

        # Observation noise
        self._R = 0.01

        # UCB parameter (adaptive)
        self.alpha = 1.0
        self._alpha_min = 0.3
        self._alpha_decay_rate = 120.0

        # Reward history for safe fallback
        self._reward_ema = [0.0 for _ in range(n_arms)]
        self._reward_alpha = 0.1

        # Identity for convenience
        self._I = np.eye(dim)

    def _blended_P(self, arm):
        """Return blended covariance: (1-p)*P_stable + p*P_volatile."""
        p = self._regime_prob[arm]
        return (1.0 - p) * self.P_stable[arm] + p * self.P_volatile[arm]

    def _blended_Q(self, arm):
        """Return blended process noise."""
        p = self._regime_prob[arm]
        return (1.0 - p) * self._Q_stable + p * self._Q_volatile

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            if any(abs(r) > 1e-10 for r in self._reward_ema):
                return int(np.argmax(self._reward_ema))
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ ctx
            P_blend = self._blended_P(a)
            var = ctx @ P_blend @ ctx
            score = mu + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Update reward EMA for fallback
        self._reward_ema[arm] = (1 - self._reward_alpha) * self._reward_ema[arm] + self._reward_alpha * reward

        # Get blended covariance and Q for this update
        P_blend = self._blended_P(arm)
        q_blend = self._blended_Q(arm)

        # Predict: inflate both covariance matrices with their respective Q
        self.P_stable[arm] = self.P_stable[arm] + self._Q_stable * self._I
        self.P_volatile[arm] = self.P_volatile[arm] + self._Q_volatile * self._I

        # Innovation
        pred = self.theta[arm] @ ctx
        innovation = reward - pred
        # Innovation variance using blended P
        S = ctx @ P_blend @ ctx + self._R
        # Kalman gain using blended P
        K = (P_blend @ ctx) / S
        # State update
        self.theta[arm] = self.theta[arm] + K * innovation

        # Covariance update for both regimes (Joseph form)
        I_KK = self._I - np.outer(K, ctx)
        self.P_stable[arm] = I_KK @ self.P_stable[arm] @ I_KK.T + self._Q_stable * self._I
        self.P_volatile[arm] = I_KK @ self.P_volatile[arm] @ I_KK.T + self._Q_volatile * self._I

        # Symmetrize
        self.P_stable[arm] = 0.5 * (self.P_stable[arm] + self.P_stable[arm].T)
        self.P_volatile[arm] = 0.5 * (self.P_volatile[arm] + self.P_volatile[arm].T)

        # Update regime probability based on innovation magnitude
        if self.count[arm] >= 2:
            norm_innov_sq = (innovation * innovation) / max(S, 1e-6)
            self._innov_ema[arm] = (1 - self._innov_alpha) * self._innov_ema[arm] + self._innov_alpha * norm_innov_sq

            # Smooth sigmoid: high innovation -> high volatile probability
            z = self._innov_ema[arm]
            sigmoid = 1.0 / (1.0 + np.exp(-self._sigmoid_steepness * (z - self._sigmoid_center)))

            # Blend new regime prob with previous (temporal smoothing)
            prev_p = self._regime_prob[arm]
            self._regime_prob[arm] = 0.7 * prev_p + 0.3 * sigmoid
        else:
            # Early: stay stable
            self._regime_prob[arm] = 0.0

        self.count[arm] += 1

        # Adaptive exploration: decay alpha over time
        self.alpha = max(self._alpha_min, 1.0 * np.exp(-self.t / self._alpha_decay_rate))
# EVOLVE-BLOCK-END
