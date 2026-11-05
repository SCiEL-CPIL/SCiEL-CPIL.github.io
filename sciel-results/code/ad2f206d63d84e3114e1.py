"""Non-stationary linear contextual bandit with adaptive forgetting and efficient updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with inverse-covariance updates and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Store inverse covariance V = A^{-1} for efficient updates
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive exploration and forgetting
        self.gamma = 0.95  # Base forgetting factor (fast adaptation per findings)
        self.gamma_min = 0.85
        self.gamma_max = 0.99
        self.C = 1.0  # Exploration parameter
        self.C_min = 0.5
        self.C_max = 3.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s

        # Track recent rewards for adaptive gamma
        self.recent_rewards = []
        self.recent_preds = []
        self.window = 50

    def _check_deadline(self):
        """Return True if we should stop exploring and exploit."""
        return time.time() > self.deadline

    def _adaptive_gamma(self):
        """Adjust gamma based on prediction errors."""
        if len(self.recent_rewards) < 10:
            return self.gamma
        errors = np.array(self.recent_rewards) - np.array(self.recent_preds)
        mse = np.mean(errors ** 2)
        # If errors are large, forget faster (lower gamma)
        # If errors are small, forget slower (higher gamma)
        target_gamma = 0.99 - 0.1 * min(mse / 10.0, 1.0)
        target_gamma = np.clip(target_gamma, self.gamma_min, self.gamma_max)
        # Smooth update
        self.gamma = 0.95 * self.gamma + 0.05 * target_gamma
        return self.gamma

    def _adaptive_C(self):
        """Adjust exploration based on recent performance."""
        if len(self.recent_rewards) < 20:
            return self.C
        # Higher C when variance is high
        var = np.var(self.recent_rewards)
        target_C = 1.0 + 0.5 * min(var, 5.0)
        target_C = np.clip(target_C, self.C_min, self.C_max)
        self.C = 0.9 * self.C + 0.1 * target_C
        return self.C

    def select(self, context):
        # Deadline guard: if close to deadline, exploit best arm
        if self._check_deadline():
            # Exploit: pick arm with highest estimated reward
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        self._adaptive_gamma()
        self._adaptive_C()

        scores = []
        for a in range(self.n):
            inv = self.V[a]
            theta = inv @ self.b[a]
            mu = theta @ context
            # UCB bonus
            bonus = self.C * np.sqrt(context @ inv @ context)
            scores.append(mu + bonus)
        scores = np.array(scores)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track for adaptive parameters
        if len(self.recent_rewards) >= self.window:
            self.recent_rewards.pop(0)
            self.recent_preds.pop(0)
        # Store prediction for error tracking
        if not hasattr(self, '_last_pred'):
            self._last_pred = {}
        self.recent_rewards.append(reward)
        if arm in self._last_pred:
            self.recent_preds.append(self._last_pred[arm])
        else:
            self.recent_preds.append(reward)  # No prediction available

        # Update inverse covariance using Sherman-Morrison
        gamma = self.gamma
        V = self.V[arm]
        b = self.b[arm]

        # Forget: V_new = V / gamma, b_new = b / gamma
        V = V / gamma
        b = b / gamma

        # Add new observation: V = V - (V x x^T V) / (1 + x^T V x)
        Vx = V @ context
        denom = 1.0 + context @ Vx
        V = V - np.outer(Vx, Vx) / denom

        # b = b + reward * context
        b = b + reward * context

        self.V[arm] = V
        self.b[arm] = b
        self.counts[arm] += 1

        # Store prediction for next error computation
        theta = self.V[arm] @ self.b[arm]
        self._last_pred[arm] = theta @ context
# EVOLVE-BLOCK-END
