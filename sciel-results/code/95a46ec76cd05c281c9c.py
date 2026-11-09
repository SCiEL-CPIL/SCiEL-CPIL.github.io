"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Multi-timescale sliding window LinUCB for non-stationary linear contextual bandits.
    
    Maintains two parallel estimators:
    - Short-window model: rapid detection of recent weight shifts
    - Long-window model: stable baseline estimation
    
    Selection uses an uncertainty-weighted blend of short and long window predictions,
    with exploration driven by the short-window uncertainty to prioritize
    immediate adaptation.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Short window: fast adaptation
        self.gamma_s = 0.90
        self.A_s = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_s = [np.zeros(dim) for _ in range(n_arms)]
        
        # Long window: stable baseline
        self.gamma_l = 0.98
        self.A_l = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_l = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB parameters
        self.alpha = 1.5   # Exploration weight for short window

    def _get_theta(self, A, b):
        """Compute theta = A^{-1} b."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def _get_width_sq(self, context, A):
        """Compute squared confidence width x^T A^{-1} x."""
        try:
            return context @ np.linalg.solve(A, context)
        except np.linalg.LinAlgError:
            return self.d

    def select(self, context):
        """Select arm based on uncertainty-weighted prediction and short-window UCB."""
        scores = []
        for a in range(self.n):
            theta_s = self._get_theta(self.A_s[a], self.b_s[a])
            theta_l = self._get_theta(self.A_l[a], self.b_l[a])
            
            mu_s = theta_s @ context
            mu_l = theta_l @ context
            
            # Compute squared widths (variances)
            w_s_sq = self._get_width_sq(context, self.A_s[a])
            w_l_sq = self._get_width_sq(context, self.A_l[a])
            
            # Uncertainty-weighted blend
            # If short window is uncertain (high w_s_sq), weight on short is low.
            # If short window is confident (low w_s_sq), weight on short is high.
            # w_s = w_l_sq / (w_s_sq + w_l_sq)
            # w_l = w_s_sq / (w_s_sq + w_l_sq)
            denom = w_s_sq + w_l_sq
            if denom < 1e-6:
                w_s = 0.5
                w_l = 0.5
            else:
                w_s = w_l_sq / denom
                w_l = w_s_sq / denom
            
            mu_blend = w_s * mu_s + w_l * mu_l
            
            # Exploration width from short window
            width_s = np.sqrt(w_s_sq)
            
            score = mu_blend + self.alpha * width_s
            scores.append(score)
        
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        """Update both short and long window models."""
        # Update short window (fast forgetting)
        self.A_s[arm] = self.gamma_s * self.A_s[arm] + np.outer(context, context)
        self.b_s[arm] = self.gamma_s * self.b_s[arm] + reward * context
        
        # Update long window (slow forgetting)
        self.A_l[arm] = self.gamma_l * self.A_l[arm] + np.outer(context, context)
        self.b_l[arm] = self.gamma_l * self.b_l[arm] + reward * context
# EVOLVE-BLOCK-END
