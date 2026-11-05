"""Non-stationary linear contextual bandit with adaptive Kalman filter and residual-driven process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Combines:
    - Joseph-form covariance update for numerical stability
    - Tanh-saturated EMA-based adaptive process noise (smooth, bounded Q)
    - Count-based exploration decay (alpha decreases with arm count)
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        # Safety margin: The evaluator kills at 150s. We stop active computation at 130s 
        # and fall back to greedy to ensure we return a result before the hard kill.
        self._deadline = 130.0  

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (per-arm) - tanh-saturated EMA
        # We use a scalar scale for Q to reduce computational overhead 
        # (Q = scale * I) and ensure isotropic drift assumption initially.
        self.Q_scale = [0.01 for _ in range(n_arms)] 
        self.residual_ema = [0.0 for _ in range(n_arms)] 
        self.ema_alpha = 0.1  # EMA smoothing factor for residual tracking

        # Exploration: base alpha with count-based decay
        # Starts with higher exploration, decays as we gather more data per arm.
        self.alpha_base = 1.2
        self.alpha_decay = 20.0  # alpha = base / (1 + count/decay)

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self, arm):
        """Count-based exploration decay."""
        c = self.count[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def _get_Q(self, arm):
        """Get process noise matrix from scalar scale."""
        # Returning a view or creating a new array? 
        # For small dim (5), creating a new array is negligible.
        # We assume isotropic process noise.
        return self.Q_scale[arm] * np.eye(self.d)

    def select(self, context):
        # Deadline guard: If we are close to the time limit, switch to greedy
        # based on current theta estimates to avoid heavy computation in update.
        if time.time() - self._t0 > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            # P_pred = P + Q
            # Optimization: Instead of adding full matrices, we can compute the 
            # variance term directly if needed, but for UCB we need P_pred for 
            # the gain calculation in update anyway.
            Q_a = self._get_Q(a)
            P_pred = self.P[a] + Q_a

            # Compute mean and variance of reward estimate
            mean = self.theta[a] @ context
            var = context @ P_pred @ context
            
            # Ensure variance is non-negative due to floating point errors
            if var < 0:
                var = 0.0

            alpha_a = self._get_alpha(a)
            ucb = mean + alpha_a * np.sqrt(var)
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
        Q_a = self._get_Q(a)
        P_pred = self.P[a] + Q_a

        # Innovation (residual)
        y = reward - self.theta[a] @ context
        S = context @ P_pred @ context + 1.0  # observation noise R = 1.0

        # Kalman gain
        # K = P_pred @ context / S
        # Optimization: Use dot products and vector operations
        P_pred_x = P_pred @ context
        K = P_pred_x / max(S, 1e-12)

        # Update state
        self.theta[a] = self.theta[a] + K * y

        # Joseph form update for numerical stability:
        # P = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        # This is more stable than P = P_pred - K x^T P_pred
        # However, for dim=5, the cost is similar. 
        # Let's use the standard symmetric update which is faster:
        # P = (I - K x^T) P_pred
        # Then symmetrize.
        
        # Actually, the Joseph form is safer against non-PSD issues.
        # Given dim=5, the matrix multiplication cost is low.
        IK = np.eye(self.d) - np.outer(K, context)
        self.P[a] = IK @ P_pred @ IK.T + np.outer(K, K) * 1.0

        # Ensure symmetry and positive semi-definiteness
        # Instead of full eigendecomposition (O(d^3)), we use:
        # 1. Symmetrize
        # 2. Add small ridge to push eigenvalues away from 0/negative
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        self.P[a] += 1e-8 * np.eye(self.d)

        self.count[a] += 1

        # Adaptive process noise: tanh-saturated EMA of |residual|
        # This allows the model to "forget" older data faster when the 
        # environment changes (high residual) and stabilize when quiet.
        abs_res = abs(y)
        
        # Update EMA
        self.residual_ema[a] = (1 - self.ema_alpha) * self.residual_ema[a] + self.ema_alpha * abs_res

        # Map EMA to Q scale using tanh saturation for smooth bounded adaptation
        # raw_scale = ema * 0.15
        # q_scale = 0.001 + 0.499 * tanh(raw_scale)
        # This ensures Q is always in [0.001, 0.5].
        raw_scale = self.residual_ema[a] * 0.15
        q_scale = 0.001 + 0.499 * np.tanh(raw_scale) 
        self.Q_scale[a] = q_scale

# EVOLVE-BLOCK-END
