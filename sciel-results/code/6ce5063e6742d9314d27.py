"""Rewritten program for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive exploration and forgetting
        self.gamma = 0.97
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.C = 1.5
        self.C_min = 0.5
        self.C_max = 3.0

        # Error tracking for adaptive gamma
        self.prediction_errors = []
        self.error_ema = 0.0
        self.error_alpha = 0.1

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0
        self.elapsed = 0.0
        self.round = 0

        # Track best arm for fallback
        self.best_arm = 0
        self.best_reward = -np.inf
        self.arm_rewards = [0.0] * n_arms
        self.arm_counts = [0] * n_arms

    def _check_deadline(self):
        self.elapsed = time.time() - self.start_time
        if self.elapsed >= self.deadline:
            return True
        return False

    def select(self, context):
        self.round += 1

        # Deadline guard: if we're close to timeout, return best known arm
        if self._check_deadline():
            return self.best_arm

        context = np.asarray(context, dtype=np.float64)

        # Compute scores with UCB
        scores = np.zeros(self.n)
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mean = theta @ context
            # Uncertainty term
            var = context @ self.V[a] @ context
            scores[a] = mean + self.C * np.sqrt(max(var, 0))

        # Adaptive C based on round and error
        # Increase exploration early, decrease later
        if self.round < 50:
            self.C = self.C_max
        elif self.round < 200:
            self.C = self.C_max - (self.C_max - 1.5) * (self.round - 50) / 150.0
        else:
            self.C = max(self.C_min, 1.5 - 0.5 * (self.round - 200) / 1000.0)

        # Add small random tie-breaker to avoid systematic bias
        scores += np.random.normal(0, 0.01, self.n)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Track rewards for fallback
        self.arm_rewards[arm] = 0.95 * self.arm_rewards[arm] + 0.05 * reward
        self.arm_counts[arm] += 1
        if self.arm_counts[arm] >= 5:
            if self.arm_rewards[arm] > self.best_reward:
                self.best_reward = self.arm_rewards[arm]
                self.best_arm = arm

        # Adaptive forgetting factor based on prediction error
        V = self.V[arm]
        b = self.b[arm]

        # Compute prediction
        theta = V @ b
        pred = theta @ context
        error = reward - pred

        # Update error EMA
        self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * abs(error)

        # Adapt gamma: larger errors -> faster forgetting (smaller gamma)
        # Map error_ema to gamma range
        if self.error_ema > 2.0:
            self.gamma = self.gamma_min
        elif self.error_ema > 1.0:
            self.gamma = self.gamma_min + 0.5 * (self.gamma_max - self.gamma_min) * (2.0 - self.error_ema)
        elif self.error_ema > 0.5:
            self.gamma = 0.95 + 0.04 * (1.0 - self.error_ema)
        else:
            self.gamma = self.gamma_max

        # Update with forgetting
        gamma = self.gamma
        V = gamma * V
        b = gamma * b

        # Sherman-Morrison update: (gamma*V + xx^T)^{-1}
        # = V - V x (x^T V x + 1/gamma)^{-1} x^T V
        # Actually: new_V = (gamma * old_V + x x^T)^{-1}
        # Using matrix inversion lemma: (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma * old_V, u = x, v = x
        # new_V = A^{-1} - A^{-1} x x^T A^{-1} / (1 + x^T A^{-1} x)
        # A^{-1} = old_V / gamma
        # new_V = old_V/gamma - (old_V/gamma) x x^T (old_V/gamma) / (1 + x^T old_V/gamma x)
        # = old_V/gamma - old_V x x^T old_V / (gamma * (1 + x^T old_V x / gamma))
        # = old_V/gamma - old_V x x^T old_V / (gamma + x^T old_V x)

        Vx = V @ context  # This is old_V @ x (before the gamma scaling above... wait)
        # Let me redo: after V = gamma * V, V is the scaled version
        # We need (V + xx^T)^{-1} where V = gamma * old_V
        # = V^{-1} - V^{-1} x x^T V^{-1} / (1 + x^T V^{-1} x)
        # V^{-1} = old_V / gamma
        V_inv = V / gamma  # This is old_V
        Vx = V_inv @ context
        denom = 1.0 + context @ Vx
        V = V_inv - np.outer(Vx, Vx) / denom

        b = b + reward * context

        self.V[arm] = V
        self.b[arm] = b
        self.counts[arm] += 1
# EVOLVE-BLOCK-END
