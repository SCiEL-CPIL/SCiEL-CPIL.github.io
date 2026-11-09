"""
Strategy: Adaptive Discounted Recursive Least Squares (RLS) with UCB.

This strategy is materially different from the current sliding-window regression in three ways:
1.  **Recursive vs. Batch**: Instead of storing a window of raw data and re-fitting via least-squares
    O(W*d^2) on every step, this uses Recursive Least Squares. It maintains a sufficient statistic
    (A matrix, b vector) and updates it in O(d^2) per step. This allows for much larger effective
    memory without computational penalty.
2.  **Exponential Discounting**: Instead of a hard window cut-off (sliding window), it uses a
    discount factor gamma. This provides a smooth "memory" where older data decays exponentially.
    This is often superior to hard windows for non-stationary environments as it avoids the
    "cliff" effect when a window shifts.
3.  **Adaptive Gamma**: The discount factor gamma is not fixed. It is adapted based on the
    prediction error (innovation). When the model predicts well, gamma stays high (long memory,
    stability). When the model makes a large error (indicating potential drift), gamma drops
    (short memory, fast adaptation). This is the core of the "Adaptive Discounted RLS" approach.
4.  **UCB Selection**: Uses Upper Confidence Bound for arm selection, which provides a
    deterministic balance between exploitation (mean reward) and exploration (uncertainty).
"""

import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin
        
        # RLS parameters
        # Initial inverse covariance matrix. Large initial value = high uncertainty = high exploration.
        # We start with a relatively tight prior to avoid excessive random exploration, 
        # but not so tight that we ignore early data.
        self.lam_init = 1.0  # Regularization for initial inverse covariance
        self.A_inv = [np.eye(dim) / self.lam_init for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Discounting parameters
        self.gamma_min = 0.8   # Minimum discount factor (fastest forgetting)
        self.gamma_max = 0.999 # Maximum discount factor (slowest forgetting)
        self.gamma = 0.95      # Current discount factor
        
        # Adaptive gamma parameters
        # If prediction error is high, we want to decrease gamma (forget faster)
        # If prediction error is low, we want to increase gamma (remember longer)
        # alpha controls the speed of adaptation
        self.alpha = 0.05
        
        # UCB parameter
        # beta controls the trade-off between exploitation and exploration
        # Too high: too much exploration. Too low: too much exploitation.
        # We start with a moderate value and might adapt it if needed, but fixed is often robust.
        self.beta = 1.5
        
        # Noise variance estimate for scaling the UCB bonus
        # In linear bandits, the uncertainty is sqrt(x^T A_inv x) * sigma
        # We estimate sigma^2 from the residuals.
        self.sigma2 = 1.0
        self.sigma2_alpha = 0.1  # Learning rate for sigma2 adaptation

    def _update_rls(self, arm, context, reward):
        """
        Update the RLS model for a specific arm.
        """
        x = context
        y = reward
        
        # Get current state
        A_inv = self.A_inv[arm]
        b = self.b[arm]
        
        # Prediction
        theta = A_inv @ b
        y_pred = theta @ x
        
        # Innovation (prediction error)
        e = y - y_pred
        
        # Update sigma2 (noise variance estimate)
        # Use an exponential moving average of the squared error
        self.sigma2 = (1 - self.sigma2_alpha) * self.sigma2 + self.sigma2_alpha * (e ** 2)
        
        # Adaptive Gamma
        # Normalize the error to get a sense of "surprise"
        # We want gamma to be low when error is large.
        # A simple heuristic: gamma = gamma_min + (gamma_max - gamma_min) * exp(-|e|/scale)
        # Or: gamma = gamma_min + (gamma_max - gamma_min) * (1 - min(|e|/threshold, 1))
        # Let's use a sigmoid-like mapping.
        # If |e| is small, gamma -> gamma_max. If |e| is large, gamma -> gamma_min.
        # scale = 1.0 is a reasonable default for unit-variance rewards.
        scale = 1.0
        error_norm = abs(e) / scale
        # Map error_norm to [0, 1] using exponential decay
        # exp(-error_norm) is 1 when e=0, and decays as e increases.
        gamma_target = self.gamma_min + (self.gamma_max - self.gamma_min) * np.exp(-error_norm)
        
        # Smoothly adapt gamma to avoid oscillations
        self.gamma = (1 - self.alpha) * self.gamma + self.alpha * gamma_target
        
        # RLS Update Step
        # A_inv_new = (1/gamma) * (A_inv_old - (A_inv_old @ x @ x^T @ A_inv_old) / (gamma + x^T @ A_inv_old @ x))
        # b_new = (1/gamma) * (b_old + A_inv_old @ x * (y - x^T @ theta_old))
        
        # Note: The standard RLS update with discounting is:
        # A_inv_t = (1/gamma) * [ A_inv_{t-1} - (A_inv_{t-1} x x^T A_inv_{t-1}) / (gamma + x^T A_inv_{t-1} x) ]
        # b_t = (1/gamma) * [ b_{t-1} + A_inv_{t-1} x (y - x^T theta_{t-1}) ]
        
        # Let's compute the terms
        A_inv_x = A_inv @ x
        x_A_inv_x = x @ A_inv_x  # scalar
        
        # Denominator for the gain
        denom = self.gamma + x_A_inv_x
        
        # Gain vector
        # K = A_inv_x / denom
        K = A_inv_x / denom
        
        # Update A_inv
        # A_inv_new = (1/gamma) * (A_inv - K @ x^T @ A_inv)
        # Note: K @ x^T @ A_inv = (A_inv @ x / denom) @ x^T @ A_inv = (A_inv @ x @ x^T @ A_inv) / denom
        # This is equivalent to the formula above.
        A_inv_new = (1.0 / self.gamma) * (A_inv - np.outer(K, x @ A_inv))
        
        # Update b
        # b_new = (1/gamma) * (b + K * e)
        b_new = (1.0 / self.gamma) * (b + K * e)
        
        # Store updated state
        self.A_inv[arm] = A_inv_new
        self.b[arm] = b_new

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n_arms):
            A_inv = self.A_inv[a]
            b = self.b[a]
            
            # Predicted mean
            theta = A_inv @ b
            mean_reward = theta @ context
            
            # Uncertainty
            # Variance of prediction: x^T A_inv x * sigma2
            x_A_inv_x = context @ (A_inv @ context)
            pred_var = x_A_inv_x * self.sigma2
            
            # UCB Score
            # Score = Mean + beta * sqrt(Variance)
            # We add a small epsilon to avoid sqrt of negative due to numerical issues
            uncertainty = np.sqrt(max(pred_var, 1e-8))
            score = mean_reward + self.beta * uncertainty
            
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self._update_rls(arm, context, reward)
        self.t += 1
