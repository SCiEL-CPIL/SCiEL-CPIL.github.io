"""
Non-stationary linear contextual bandit policy.
Uses a single global linear model with arm identity as explicit one-hot features,
allowing cross-arm correlation learning and shared context representation.
Adaptive exponential forgetting for non-stationary tracking.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms  # augmented dimension: context + one-hot arm
        
        # Single global model: A (precision matrix) and b (moment vector)
        # Features are [context, one_hot_arm]
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        
        # Adaptive forgetting parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.78   # Faster forgetting for regime changes
        self.max_lambda = 0.995  # Slow forgetting when stable
        
        # Error tracking (global)
        self.error_buffer = []
        self.error_window_size = 30
        
        # UCB exploration parameter
        self.alpha = 2.0
        
        # Track last predictions for error computation
        self._last_preds = [0.0] * n_arms
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0
        
        # Track rounds
        self.total_rounds = 0
        
        # Global reward statistics for adaptive alpha
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        
        # Arm counts for potential exploration decay
        self.arm_counts = [0] * n_arms

    def _get_lambda(self):
        """
        Calculate the adaptive forgetting factor based on recent prediction errors.
        """
        if not self.error_buffer:
            return self.base_lambda

        recent = self.error_buffer[-self.error_window_size:]
        mean_err = np.mean(recent)

        err_scale = 0.15
        normalized_err = mean_err / err_scale

        if normalized_err <= 0.05:
            return self.max_lambda
        elif normalized_err >= 2.0:
            return self.min_lambda
        else:
            t = (normalized_err - 0.05) / (2.0 - 0.05)
            t = t * t * (3 - 2 * t)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _get_alpha(self):
        """
        Adaptively adjust the exploration parameter based on global performance.
        """
        if self.total_rounds < 50:
            return self.alpha

        mean_reward = self.reward_sum / self.total_rounds
        var_reward = max(0, self.reward_sq_sum / self.total_rounds - mean_reward**2)
        std_reward = np.sqrt(var_reward)

        base_alpha = 1.5
        if std_reward > 0.3:
            alpha = base_alpha + 1.5
        elif std_reward > 0.1:
            alpha = base_alpha + 0.5
        else:
            alpha = base_alpha - 0.5

        return max(0.5, min(4.0, alpha))

    def _augment(self, context, arm):
        """Create augmented feature vector [context, one_hot_arm]."""
        x = np.asarray(context)
        aug = np.zeros(self.d_aug)
        aug[:self.d] = x
        aug[self.d + arm] = 1.0
        return aug

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))
        
        context = np.asarray(context)
        scores = []
        alpha = self._get_alpha()
        
        try:
            inv_A = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            return int(np.random.randint(self.n))
        
        for a in range(self.n):
            # Build augmented feature for this arm
            aug = np.zeros(self.d_aug)
            aug[:self.d] = context
            aug[self.d + a] = 1.0
            
            # Mean prediction
            theta = inv_A @ self.b
            mu = theta @ aug
            
            # Store prediction for error tracking
            self._last_preds[a] = mu
            
            # Variance term
            var = aug @ inv_A @ aug
            std = np.sqrt(max(0, var))
            
            # UCB score
            score = mu + alpha * std
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() > self._deadline:
            return
        
        context = np.asarray(context)
        
        # Update global reward statistics
        self.total_rounds += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward
        
        # Compute prediction error
        pred = self._last_preds[arm]
        error = abs(reward - pred)
        
        # Append error to buffer
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.error_window_size * 2:
            self.error_buffer = self.error_buffer[-self.error_window_size:]
        
        # Update global model with adaptive forgetting
        lam = self._get_lambda()
        
        aug = np.zeros(self.d_aug)
        aug[:self.d] = context
        aug[self.d + arm] = 1.0
        
        self.A = lam * self.A + np.outer(aug, aug)
        self.b = lam * self.b + reward * aug
        
        # Update arm count
        self.arm_counts[arm] += 1
        
        # Periodic regularization to prevent numerical issues
        if self.total_rounds % 100 == 0:
            self.A += 0.001 * np.eye(self.d_aug)
