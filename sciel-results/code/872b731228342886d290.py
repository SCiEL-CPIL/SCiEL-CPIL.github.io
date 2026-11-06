"""Non-stationary linear contextual bandit with global model and arm-specific biases."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - A single global linear model for context features
    - Arm-specific bias terms (one-hot encoding approach)
    - Recursive Least Squares (RLS) with exponential forgetting
    - Adaptive exploration based on prediction uncertainty
    - Deadline guard to prevent timeout
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.total_dim = dim + n_arms  # context + one-hot arm indicators
        
        # Global RLS state
        # We model: reward = [context, arm_one_hot] @ theta
        self.P = np.eye(self.total_dim) / 1.0  # Inverse covariance matrix
        self.theta = np.zeros(self.total_dim)
        
        # Forgetting factor
        self.lambda_ = 0.98
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Track last update time per arm for time-aware exploration
        self.last_played = np.full(n_arms, -100)
        
        # Adaptive forgetting: track recent prediction errors
        self.error_window = []
        self.error_window_size = 20

    def _augment(self, context, arm):
        """Create augmented feature vector: [context, one_hot(arm)]"""
        feat = np.zeros(self.total_dim)
        feat[:self.d] = context
        feat[self.d + arm] = 1.0
        return feat

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        """Select arm using UCB with global model uncertainty."""
        if self._check_deadline():
            # Near deadline: exploit best known arm
            best_arm = np.argmax(self.theta[self.d:self.d + self.n])
            return int(best_arm)
        
        scores = []
        context = np.asarray(context, dtype=np.float64)
        
        for a in range(self.n):
            feat = self._augment(context, a)
            
            # Predicted reward
            pred = feat @ self.theta
            
            # Uncertainty: sqrt(feat^T P feat)
            uncertainty = np.sqrt(feat @ self.P @ feat)
            
            # Time-aware exploration bonus: increase if arm hasn't been played recently
            time_since_played = self.t - self.last_played[a]
            time_bonus = 0.1 * np.log1p(time_since_played)
            
            # UCB score
            score = pred + 0.5 * uncertainty + time_bonus
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update RLS model with new observation."""
        if self._check_deadline():
            return
        
        context = np.asarray(context, dtype=np.float64)
        feat = self._augment(context, arm)
        
        # RLS update
        P_feat = self.P @ feat
        denom = 1.0 + feat @ P_feat
        gain = P_feat / denom
        
        # Prediction error
        pred = feat @ self.theta
        error = reward - pred
        
        # Update theta
        self.theta = self.theta + gain * error
        
        # Update P with forgetting
        self.P = (self.P - np.outer(gain, P_feat)) / self.lambda_
        
        # Track prediction error for adaptive forgetting
        self.error_window.append(abs(error))
        if len(self.error_window) > self.error_window_size:
            self.error_window.pop(0)
        
        # Adapt forgetting factor based on recent errors
        if len(self.error_window) >= 5:
            mean_error = np.mean(self.error_window)
            # Higher errors -> more forgetting (smaller lambda)
            self.lambda_ = np.clip(0.995 - 0.02 * mean_error, 0.95, 0.995)
        
        self.last_played[arm] = self.t
        self.t += 1
# EVOLVE-BLOCK-END
