"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Non-stationary LinUCB with adaptive forgetting and soft-reset on large errors."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Forgetting factor parameters
        self.base_forgetting = 0.99
        self.min_forgetting = 0.95
        self.max_forgetting = 0.999
        
        # Error tracking for adaptive forgetting
        self.error_history = [deque(maxlen=20) for _ in range(n_arms)]
        self.consecutive_large_errors = [0 for _ in range(n_arms)]
        self.error_threshold = 1.5
        
        # Exploration parameter
        self.alpha = 1.0
        self.min_alpha = 0.3
        self.max_alpha = 2.0
        
        # Time tracking
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit
        self.round = 0

    def _update_theta(self, arm):
        """Update theta and A_inv using Sherman-Morrison or direct inversion."""
        try:
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.linalg.pinv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]

    def _adaptive_forgetting(self, arm, prediction_error):
        """Adapt forgetting factor based on recent prediction errors."""
        abs_error = abs(prediction_error)
        self.error_history[arm].append(abs_error)
        
        if abs_error > self.error_threshold:
            self.consecutive_large_errors[arm] += 1
        else:
            self.consecutive_large_errors[arm] = max(0, self.consecutive_large_errors[arm] - 1)
        
        # Increase forgetting (faster decay) when we see consecutive large errors
        base = self.base_forgetting
        if self.consecutive_large_errors[arm] > 2:
            # Soft reset: use higher forgetting rate
            return self.min_forgetting
        elif self.consecutive_large_errors[arm] > 0:
            # Moderate increase in forgetting
            return (base + self.min_forgetting) / 2
        
        # If errors are consistently small, decrease forgetting (more memory)
        if len(self.error_history[arm]) >= 10:
            recent_errors = list(self.error_history[arm])[-10:]
            if np.mean(recent_errors) < 0.5:
                return self.max_forgetting
        
        return base

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return best arm based on current theta
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))
        
        self.round += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive alpha: start higher, decrease over time but keep some exploration
        progress = min(self.round / 1000.0, 1.0)
        current_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * progress
        
        scores = []
        for a in range(self.n):
            mean_reward = self.theta[a] @ context
            # Uncertainty term
            uncertainty = np.sqrt(max(0, context @ self.A_inv[a] @ context))
            score = mean_reward + current_alpha * uncertainty
            scores.append(score)
        
        # Add small random noise for exploration in early rounds
        if self.round < 50:
            scores = [s + np.random.randn() * 0.1 for s in scores]
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Compute prediction error
        predicted = self.theta[arm] @ context
        prediction_error = reward - predicted
        
        # Get adaptive forgetting factor
        forgetting = self._adaptive_forgetting(arm, prediction_error)
        
        # Update A and b with forgetting
        self.A[arm] = forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = forgetting * self.b[arm] + reward * context
        
        # Update A_inv using Sherman-Morrison formula
        A_inv_old = self.A_inv[arm]
        x = context
        f = forgetting
        
        # Compute x^T A^{-1} x
        A_inv_x = A_inv_old @ x
        denom = 1.0 + (1.0 / f) * (x @ A_inv_x)
        
        if abs(denom) < 1e-10:
            # Fallback to direct inversion
            self.A_inv[arm] = np.linalg.pinv(self.A[arm])
        else:
            self.A_inv[arm] = (1.0 / f) * A_inv_old - (1.0 / (f * f * denom)) * np.outer(A_inv_x, A_inv_x)
        
        # Symmetrize to reduce numerical errors
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
        
        # Update theta
        self._update_theta(arm)
        
        # Occasionally re-orthogonalize to prevent drift
        if self.round % 100 == 0:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.linalg.pinv(self.A[arm])
            self._update_theta(arm)
# EVOLVE-BLOCK-END
