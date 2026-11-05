# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Exponentially-weighted linear bandit with Sherman-Morrison rank-1 updates and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as identity (ridge) for each arm
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Precompute inverses for Sherman-Morrison updates
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting factor
        self.gamma = 0.99
        # Exploration parameter
        self.alpha = 1.0
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s
        self.round_count = 0

    def _sherman_morrison_update(self, arm, context, reward):
        """Update A_inv and theta using Sherman-Morrison formula after rank-1 update."""
        x = context
        # A_new = gamma * A + x x^T
        # We maintain A_inv such that A_inv @ A = I
        # After update: A_new = gamma * A + outer(x, x)
        # A_new_inv = (gamma * A + x x^T)^{-1}
        # Using Sherman-Morrison: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_new = gamma * A + x x^T = gamma (A + (1/gamma) x x^T)
        # So A_new^{-1} = (1/gamma) (A + (1/gamma) x x^T)^{-1}
        # = (1/gamma) [A^{-1} - A^{-1} (x/gamma) (x/gamma)^T A^{-1} / (1 + (x/gamma)^T A^{-1} (x/gamma))]
        
        inv = self.A_inv[arm]
        x_over_g = x / self.gamma
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse
            self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        # A_new^{-1} = (1/gamma) * (inv - np.outer(Ax, Ax) / denom)
        new_inv = (inv - np.outer(Ax, Ax) / denom) / self.gamma
        self.A_inv[arm] = new_inv
        
        # Update b: b_new = gamma * b + reward * x
        self.b[arm] = self.gamma * self.b[arm] + reward * x
        # Update theta = A_new^{-1} @ b_new
        self.theta[arm] = new_inv @ self.b[arm]
        
        # Also update A for potential fallback use
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return a deterministic arm based on context hash to avoid exceptions
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.A_inv[a]
            mean = theta @ context
            # UCB bonus
            var = context @ inv @ context
            bonus = self.alpha * np.sqrt(max(var, 0.0))
            scores.append(mean + bonus)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Adaptive forgetting: increase gamma (slower forgetting) when prediction is good,
        # decrease gamma (faster forgetting) when prediction error is large
        pred = self.theta[arm] @ context
        pred_err = abs(reward - pred)
        # Smoothly adjust gamma: higher error -> lower gamma (faster adaptation)
        # Target gamma range: [0.90, 0.999]
        target_gamma = 0.999 - 0.099 * np.tanh(pred_err / 2.0)
        self.gamma = 0.95 * self.gamma + 0.05 * target_gamma
        self.gamma = np.clip(self.gamma, 0.90, 0.999)
        
        # Perform the update
        self._sherman_morrison_update(arm, context, reward)
        
        # Adjust exploration: reduce alpha over time but keep some exploration
        self.alpha = max(0.5, self.alpha * 0.9999)
# EVOLVE-BLOCK-END
