"""Non-stationary linear contextual bandit with hybrid linear + contextual-free approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm discounted least squares for linear model
    - Per-arm EMA of rewards as a contextual-free baseline
    - Blended score: (1-w)*linear_prediction + w*ema_reward
    - Adaptive blend weight based on prediction reliability
    - Thompson Sampling on the linear component for exploration
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0

        # Inverse covariance matrices for linear model
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # EMA of rewards per arm (contextual-free signal)
        self.ema_rewards = [0.0] * n_arms
        self.ema_alpha = 0.15  # fairly aggressive EMA for non-stationarity

        # Forgetting factor for linear model
        self.gamma = 0.90

        # Track prediction errors to adapt blend weight
        self.pred_errors = [0.0] * n_arms
        self.linear_confidence = [0.0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                lin_score = theta @ ctx
                c = self.counts[a]
                if c > 5:
                    # Use blend based on confidence
                    conf = self.linear_confidence[a]
                    w = max(0.0, 1.0 - conf) * 0.4
                    scores.append((1.0 - w) * lin_score + w * self.ema_rewards[a])
                else:
                    scores.append(lin_score)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta_hat = self.V[a] @ self.b[a]

            # Thompson Sampling on linear component
            try:
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                theta_sample = theta_hat

            lin_score = theta_sample @ ctx

            # Contextual-free EMA score
            c = self.counts[a]
            if c == 0:
                ema_score = 0.0
            else:
                ema_score = self.ema_rewards[a]

            # Blend: trust linear more as it becomes reliable
            if c < 10:
                w_ema = 0.5  # heavy exploration via EMA for new arms
            else:
                # Confidence based on recent prediction accuracy
                conf = self.linear_confidence[a]
                # If linear model predicts well, trust it more
                w_ema = max(0.0, 1.0 - conf) * 0.3

            score = (1.0 - w_ema) * lin_score + w_ema * ema_score
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        self.counts[arm] += 1

        # Update EMA
        if self.counts[arm] == 1:
            self.ema_rewards[arm] = reward
        else:
            self.ema_rewards[arm] = (1.0 - self.ema_alpha) * self.ema_rewards[arm] + self.ema_alpha * reward

        # Compute prediction error for confidence tracking
        theta_hat = self.V[arm] @ self.b[arm]
        predicted = theta_hat @ ctx
        pred_err = abs(reward - predicted)

        # Track running prediction error
        self.pred_errors[arm] = 0.9 * self.pred_errors[arm] + 0.1 * pred_err

        # Update linear confidence: lower if errors are high
        # Normalize: if pred_err is small relative to reward scale, confidence is high
        err_scale = max(self.pred_errors[arm], 1e-6)
        confidence = max(0.0, min(1.0, 1.0 - err_scale / 5.0))
        self.linear_confidence[arm] = 0.95 * self.linear_confidence[arm] + 0.05 * confidence

        # Update linear model with forgetting
        gamma = self.gamma
        V_old = self.V[arm]
        Vx = V_old @ ctx
        xVx = ctx @ Vx
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d)

        self.b[arm] = gamma * self.b[arm] + reward * ctx
# EVOLVE-BLOCK-END
