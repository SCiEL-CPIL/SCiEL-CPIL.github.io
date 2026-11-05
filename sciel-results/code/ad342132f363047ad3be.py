"""Non-stationary linear contextual bandit policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit using:
    - Exponentially weighted least squares (forgetting factor)
    - Adaptive exploration via count-based alpha decay
    - Tanh-saturated reward EMA for bounded process noise adaptation
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.99  # Forgetting factor for non-stationarity
        self.lam = 0.1     # Regularization
        
        # Per-arm state
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = np.zeros(n_arms)
        
        # Adaptive exploration: start high, decay with visits
        self.alpha_base = 1.0
        self.alpha_min = 0.3
        
        # Tanh-saturated EMA of rewards for bounded process noise tracking
        self.reward_ema = np.zeros(n_arms)
        self.reward_ema_alpha = 0.1  # EMA smoothing
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin before 120s limit

    def _get_alpha(self, arm):
        """Adaptive exploration: decays with visit count."""
        c = self.counts[arm]
        # Smooth decay: alpha = alpha_min + (alpha_base - alpha_min) * exp(-c / tau)
        tau = 50.0
        return self.alpha_min + (self.alpha_base - self.alpha_min) * np.exp(-c / tau)

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return best arm based on current estimates
            scores = []
            for a in range(self.n):
                try:
                    inv = np.linalg.inv(self.A[a])
                    theta = inv @ self.b[a]
                    scores.append(theta @ context)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64).flatten()
        scores = []
        
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mu = theta @ context
                # Variance term
                var = context @ inv @ context
                alpha = self._get_alpha(a)
                # Add tanh-saturated EMA adjustment for process noise
                ema_adjust = 0.1 * np.tanh(self.reward_ema[a])
                score = mu + alpha * np.sqrt(max(var, 1e-10)) + ema_adjust
            except np.linalg.LinAlgError:
                score = 0.0
            
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)
        
        # Update exponentially weighted statistics
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update visit count
        self.counts[arm] += 1
        
        # Update tanh-saturated reward EMA
        self.reward_ema[arm] = (1 - self.reward_ema_alpha) * self.reward_ema[arm] + self.reward_ema_alpha * reward

# EVOLVE-BLOCK-END
