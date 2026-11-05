"""Non-stationary linear contextual bandit with model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm discounted least squares (no intercept augmentation)
    - Thompson Sampling blended with EMA reward for exploration
    - Prediction-error-driven adaptive forgetting
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0

        # Inverse covariance matrices V = A^{-1} for each arm
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive forgetting parameters
        self.base_gamma = 0.93
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        self.prediction_errors = [0.0] * n_arms

        # EMA of rewards per arm (dual-signal exploration)
        self.ema_rewards = [0.0] * n_arms
        self.ema_alpha = 0.1  # EMA decay rate

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm, prediction_error):
        """Adapt forgetting factor based on prediction error."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # Track running average of prediction errors
        self.prediction_errors[arm] = 0.9 * self.prediction_errors[arm] + 0.1 * prediction_error

        pe = self.prediction_errors[arm]
        # Higher prediction error -> faster forgetting (lower gamma)
        pe_scale = min(pe, 5.0)
        gamma = self.base_gamma - 0.08 * pe_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            # Greedy fallback: pick arm with highest mean estimate
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ ctx)
            return int(np.argmax(scores))

        # Dual-signal exploration: blend Thompson Sampling with EMA reward
        # TS score: sample from posterior N(theta_hat, V)
        # EMA score: robust baseline from reward history
        # Blend: score = (1 - w) * ts_score + w * ema_score
        # w increases with count to trust the model more over time
        scores = []
        for a in range(self.n):
            theta_hat = self.V[a] @ self.b[a]
            
            # Thompson Sampling: sample from N(theta_hat, V)
            try:
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                theta_sample = theta_hat

            ts_score = theta_sample @ ctx

            # Blend with EMA reward
            c = self.counts[a]
            if c > 0:
                # Weight for EMA decreases as we trust the model more
                # After 20 pulls, we rely mostly on the model
                w_ema = max(0.0, 1.0 - c / 20.0) * 0.5
            else:
                w_ema = 0.5  # High exploration for unseen arms

            score = (1.0 - w_ema) * ts_score + w_ema * self.ema_rewards[a]
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Compute prediction error before update
        theta_hat = self.V[arm] @ self.b[arm]
        predicted = theta_hat @ ctx
        prediction_error = abs(reward - predicted)

        # Track statistics
        self.counts[arm] += 1

        # Update EMA reward
        if self.counts[arm] == 1:
            self.ema_rewards[arm] = reward
        else:
            self.ema_rewards[arm] = (1.0 - self.ema_alpha) * self.ema_rewards[arm] + self.ema_alpha * reward

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm, prediction_error)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        # = (1/gamma) * V_old - (1/gamma) * V_old x x^T V_old / (gamma + x^T V_old x)

        V_old = self.V[arm]
        Vx = V_old @ ctx
        xVx = ctx @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * ctx
# EVOLVE-BLOCK-END
