"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """LinUCB with adaptive forgetting factor and maintained inverse."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Maintain A^{-1} directly for O(d^2) updates instead of O(d^3) inversions
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor
        self.gamma = 0.98
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.error_ema = 0.0
        self.error_alpha = 0.15
        
        # Exploration parameter
        self.C = 0.5
        self.round_count = 0

    def select(self, context):
        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            scores.append(theta @ context + self.C * np.sqrt(context @ inv @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        
        # Adaptive gamma: increase discounting when error is high (non-stationary)
        if self.round_count > 10:
            # Estimate current prediction error for the selected arm
            inv = self.A_inv[arm]
            theta = inv @ self.b[arm]
            pred = theta @ context
            err = abs(reward - pred)
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * err
            
            # Map error to gamma: high error -> lower gamma (faster forgetting)
            # Normalize error to [0, 1] range approximately
            norm_err = min(err / (1.0 + self.error_ema), 1.0)
            self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * norm_err
        else:
            self.gamma = 0.99
        
        # Update b with forgetting
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update A^{-1} using Sherman-Morrison with forgetting
        # A_new = gamma * A_old + x x^T
        # We maintain V = A^{-1}
        # V_new = (gamma * A_old + x x^T)^{-1}
        # Using Woodbury: (gamma * A + x x^T)^{-1} = (1/gamma) * A^{-1} - (1/gamma^2) * A^{-1} x x^T A^{-1} / (1/gamma + x^T A^{-1} x)
        V = self.A_inv[arm]
        g = self.gamma
        x = context
        Vx = V @ x
        denom = (1.0 / g) + x @ Vx
        # V_new = (1/g) * V - (1/g^2) * Vx * Vx^T / denom
        self.A_inv[arm] = (1.0 / g) * V - (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        # Symmetrize to avoid numerical drift
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
# EVOLVE-BLOCK-END
