# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive discounted least-squares with UCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0  # safety margin before 120s kill

        # Per-arm state
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]

        # Adaptive forgetting
        self.base_gamma = 0.92
        self.gamma_min = 0.70
        self.gamma = self.base_gamma
        self.error_ema = [0.0] * n_arms
        self.ema_alpha = 0.35

        # Exploration
        self.C = 1.5

        # Reward tracking for adaptive C
        self.reward_history = []
        self.reward_ema = 0.0
        self.reward_ema_alpha = 0.1

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return greedy based on current estimates
            scores = []
            for a in range(self.n):
                theta = self.inv_A[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            v = context @ self.inv_A[a] @ context
            bonus = self.C * np.sqrt(v)
            scores.append(mu + bonus)

        scores = np.array(scores)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update reward EMA
        if self.reward_ema == 0.0 and len(self.reward_history) == 0:
            self.reward_ema = reward
        else:
            self.reward_ema = (1 - self.reward_ema_alpha) * self.reward_ema + self.reward_ema_alpha * reward

        # Track error for adaptive forgetting
        theta = self.inv_A[arm] @ self.b[arm]
        pred = theta @ context
        error = reward - pred
        abs_error = abs(error)
        self.error_ema[arm] = (1 - self.ema_alpha) * self.error_ema[arm] + self.ema_alpha * abs_error

        # Adaptive forgetting factor: lower gamma when error is high
        max_err = max(self.error_ema) if self.error_ema else 0.0
        # Scale: if error is large, forget more aggressively
        err_scale = min(max_err / (abs(self.reward_ema) + 1e-8), 2.0)
        self.gamma = max(self.gamma_min, self.base_gamma - 0.15 * err_scale)

        # Update per-arm state with discounting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Update inverse using Sherman-Morrison or direct inversion
        # For small dim, direct inversion is fine; for larger, use SM update
        # Use SM update for efficiency: (A + xx^T)^{-1} = A^{-1} - A^{-1} x x^T A^{-1} / (1 + x^T A^{-1} x)
        # But we have gamma * A + xx^T, not A + xx^T.
        # Let A_new = gamma * A_old + xx^T
        # A_new^{-1} = (gamma * A_old + xx^T)^{-1}
        # = (gamma * (A_old + (1/gamma) xx^T))^{-1}
        # = (1/gamma) * (A_old + (1/gamma) xx^T)^{-1}
        # Use SM on (A_old + (1/gamma) xx^T):
        # (A + alpha xx^T)^{-1} = A^{-1} - alpha * A^{-1} x x^T A^{-1} / (1 + alpha x^T A^{-1} x)
        # where A = A_old, alpha = 1/gamma
        # So A_new^{-1} = (1/gamma) * [A_old^{-1} - (1/gamma) * A_old^{-1} x x^T A_old^{-1} / (1 + (1/gamma) x^T A_old^{-1} x)]

        inv_old = self.inv_A[arm]
        x = context
        gamma_val = self.gamma

        # Compute x^T A^{-1} x
        Ax = inv_old @ x
        denom = 1.0 + (1.0 / gamma_val) * (x @ Ax)

        if abs(denom) < 1e-12:
            # Fallback to direct inversion
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        else:
            # SM update
            # (1/gamma) * [inv_old - (1/gamma) * inv_old @ x @ x^T @ inv_old / denom]
            outer = np.outer(Ax, x)  # A^{-1} x x^T
            correction = (1.0 / gamma_val) * outer / denom
            self.inv_A[arm] = (1.0 / gamma_val) * (inv_old - correction)

        # Symmetrize to prevent drift
        self.inv_A[arm] = (self.inv_A[arm] + self.inv_A[arm].T) / 2.0
# EVOLVE-BLOCK-END
