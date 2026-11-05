"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit with adaptive forgetting and soft-reset."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 0.1  # regularization
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        
        # Forgetting factor - start high, adapt based on error
        self.gamma = 0.95
        self.base_gamma = 0.95
        
        # Error tracking for soft-reset
        self.recent_errors = [[] for _ in range(n_arms)]
        self.error_window = 10
        
        # Exploration parameter
        self.alpha = 1.0
        self.base_alpha = 1.0
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = 150.0  # seconds
        
        # Track per-arm performance for adaptive alpha
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.reward_window = 20

    def _check_deadline(self):
        """Return True if we should stop computing and return a safe action."""
        return (time.time() - self.start_time) > self.deadline - 5.0

    def _update_theta(self, arm):
        """Update theta and A_inv for a given arm."""
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

    def _soft_reset(self, arm):
        """Inflate covariance to accelerate recovery from weight shifts."""
        # Inflate the covariance matrix to reduce confidence
        self.A[arm] = 0.5 * self.A[arm] + self.lam * np.eye(self.d)
        self.b[arm] = 0.5 * self.b[arm]
        self._update_theta(arm)

    def select(self, context):
        # Deadline guard
        if self._check_deadline():
            # Return the arm with highest estimated reward (greedy fallback)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted mean
            pred = self.theta[a] @ context
            # Uncertainty term
            var = context @ self.A_inv[a] @ context
            ucb = pred + self.alpha * np.sqrt(var)
            scores.append(ucb)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Track recent errors for this arm
        self.recent_errors[arm].append(abs(error))
        if len(self.recent_errors[arm]) > self.error_window:
            self.recent_errors[arm].pop(0)
        
        # Track rewards for adaptive alpha
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > self.reward_window:
            self.arm_rewards[arm].pop(0)
        
        # Soft reset if consecutive large errors
        if len(self.recent_errors[arm]) >= self.error_window:
            avg_error = np.mean(self.recent_errors[arm])
            if avg_error > 2.0:  # Threshold for large errors
                self._soft_reset(arm)
                self.recent_errors[arm] = []
                return
        
        # Apply forgetting factor
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update theta and inverse
        self._update_theta(arm)
        
        # Adapt gamma based on error magnitude
        if abs(error) > 1.5:
            # Large error: increase forgetting (lower gamma) to adapt faster
            self.gamma = max(0.85, self.gamma - 0.005)
        elif abs(error) < 0.3:
            # Small error: decrease forgetting (higher gamma) for stability
            self.gamma = min(0.98, self.gamma + 0.002)
        
        # Adapt alpha based on recent reward variance
        if len(self.arm_rewards[arm]) >= 5:
            recent_var = np.var(self.arm_rewards[arm])
            # If rewards are stable, reduce exploration; if volatile, increase
            target_alpha = 0.5 + min(1.5, recent_var)
            self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
# EVOLVE-BLOCK-END
