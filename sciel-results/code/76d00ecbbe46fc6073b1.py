<<<<<<< SEARCH
"""
Non-stationary linear contextual bandit policy.
Combines per-arm linear models (structural fit to r = x^T theta_a)
with a shared global context model for robust variance estimation,
and adaptive forgetting based on per-arm prediction errors.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm linear models: A (precision matrix) and b (moment vector)
        # This matches the likely true reward generation r = x^T theta_a
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Global model for shared context understanding (helps with variance)
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)

        # Adaptive forgetting parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.70   # More aggressive forgetting when errors are high
        self.max_lambda = 0.995  # Very slow forgetting when errors are low

        # Error tracking per arm
        self.error_buffers = [[] for _ in range(n_arms)]
        self.error_window_size = 30

        # UCB exploration parameter
        self.alpha = 2.0

        # Track last predictions for error computation
        self._last_preds = [0.0] * n_arms

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # Safety margin before 150s kill

        # Track number of rounds per arm for adaptive alpha
        self.arm_counts = [0] * n_arms
        self.total_rounds = 0

        # Global reward statistics for adaptive alpha
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def _get_lambda(self, arm):
        """
        Calculate the adaptive forgetting factor for a specific arm
        based on recent prediction errors.
        """
        buf = self.error_buffers[arm]
        if not buf:
            return self.base_lambda

        # Use the mean absolute error in the recent window
        recent = buf[-self.error_window_size:]
        mean_err = np.mean(recent)

        # Normalize error with a soft cap
        err_scale = 0.15
        normalized_err = mean_err / err_scale

        # Use a smoother mapping
        if normalized_err <= 0.05:
            return self.max_lambda
        elif normalized_err >= 2.0:
            return self.min_lambda
        else:
            # Smooth interpolation using a sigmoid-like function
            t = (normalized_err - 0.05) / (2.0 - 0.05)
            # Apply a smooth step function
            t = t * t * (3 - 2 * t)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _get_alpha(self):
        """
        Adaptively adjust the exploration parameter based on global performance.
        """
        # If we have enough data, adjust alpha based on reward variance
        if self.total_rounds < 50:
            return self.alpha

        # Estimate reward variance
        mean_reward = self.reward_sum / self.total_rounds
        var_reward = max(0, self.reward_sq_sum / self.total_rounds - mean_reward**2)
        std_reward = np.sqrt(var_reward)

        # If rewards are highly variable, increase exploration
        # If rewards are stable, decrease exploration
        base_alpha = 1.5
        if std_reward > 0.3:
            alpha = base_alpha + 1.5  # More exploration for high variance
        elif std_reward > 0.1:
            alpha = base_alpha + 0.5
        else:
            alpha = base_alpha - 0.5  # Less exploration for low variance

        # Clamp alpha
        return max(0.5, min(4.0, alpha))

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context)
        scores = []

        # Use adaptive alpha
        alpha = self._get_alpha()

        for a in range(self.n):
            # Compute theta and UCB
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]

                # Mean prediction
                mu = theta @ context

                # Store prediction for error tracking
                self._last_preds[a] = mu

                # Variance term: use per-arm precision for uncertainty
                var = context @ inv_A @ context
                std = np.sqrt(max(0, var))

                # UCB score
                score = mu + alpha * std
                scores.append(score)
            except np.linalg.LinAlgError:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)

        # Deadline check
        if time.time() > self._deadline:
            return

        # Update global reward statistics
        self.total_rounds += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward

        # 1. Compute prediction error
        pred = self._last_preds[arm]
        error = abs(reward - pred)

        # Append error to buffer
        self.error_buffers[arm].append(error)

        # Keep buffer size limited
        if len(self.error_buffers[arm]) > self.error_window_size * 2:
            self.error_buffers[arm] = self.error_buffers[arm][-self.error_window_size:]

        # 2. Update per-arm model with adaptive forgetting
        lam = self._get_lambda(arm)

        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        # 3. Update global model with a fixed moderate forgetting rate
        # This helps maintain a shared understanding of the context space
        lam_global = 0.99
        self.A_global = lam_global * self.A_global + np.outer(context, context)
        self.b_global = lam_global * self.b_global + reward * context

        # 4. Update arm count
        self.arm_counts[arm] += 1

        # 5. Periodic regularization to prevent numerical issues
        # Add small regularization to keep matrices well-conditioned
        if self.total_rounds % 100 == 0:
            for a in range(self.n):
                self.A[a] += 0.001 * np.eye(self.d)
            self.A_global += 0.001 * np.eye(self.d)
=======
"""
Non-stationary linear contextual bandit policy.
Uses a single global linear model with arm identity as explicit features,
allowing inter-arm correlation learning and shared context representation.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Total feature dimension: original context + one-hot arm indicators
        self.d_aug = dim + n_arms

        # Single global model: A (precision matrix) and b (moment vector)
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)

        # Adaptive forgetting parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.75
        self.max_lambda = 0.995

        # Error tracking
        self.error_buffer = []
        self.error_window_size = 30

        # UCB exploration parameter
        self.alpha = 1.8

        # Track last predictions for error computation
        self._last_preds = [0.0] * n_arms

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

        # Total rounds
        self.total_rounds = 0

        # Global reward statistics for adaptive alpha
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def _augment_context(self, context, arm):
        """
        Augment context with one-hot arm indicator.
        """
        aug = np.zeros(self.d_aug)
        aug[:self.d] = context
        aug[self.d + arm] = 1.0
        return aug

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
            alpha = base_alpha + 1.0
        elif std_reward > 0.1:
            alpha = base_alpha + 0.3
        else:
            alpha = base_alpha - 0.3

        return max(0.5, min(4.0, alpha))

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context)
        alpha = self._get_alpha()

        # Compute inverse of A once
        try:
            inv_A = np.linalg.inv(self.A)
            theta = inv_A @ self.b
        except np.linalg.LinAlgError:
            return int(np.random.randint(self.n))

        scores = []
        for a in range(self.n):
            aug = self._augment_context(context, a)
            mu = theta @ aug
            self._last_preds[a] = mu

            var = aug @ inv_A @ aug
            std = np.sqrt(max(0, var))

            score = mu + alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)

        # Deadline check
        if time.time() > self._deadline:
            return

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

        # Update model with adaptive forgetting
        lam = self._get_lambda()

        aug = self._augment_context(context, arm)
        self.A = lam * self.A + np.outer(aug, aug)
        self.b = lam * self.b + reward * aug

        # Periodic regularization to prevent numerical issues
        if self.total_rounds % 100 == 0:
            self.A += 0.001 * np.eye(self.d_aug)
>>>>>>> REPLACE
BET: {mechanism: "Replaces per-arm models with a single global model using one-hot arm features, enabling shared context learning and inter-arm correlation, which should improve sample efficiency in non-stationary settings", q: 0.72, delta: [0.05, 0.25]}
