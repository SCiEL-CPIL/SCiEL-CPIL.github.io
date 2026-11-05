"""
Non-stationary linear contextual bandit policy using Global Thompson Sampling.

Strategy:
1. Use a single shared covariance matrix (A) and mean vector (b) for all arms.
   This assumes the context features are relevant across arms and allows
   for a more stable and computationally efficient uncertainty estimate.
2. Use Thompson Sampling: sample a parameter vector theta from the posterior
   N(theta_hat, A^{-1}) and select the arm that maximizes theta @ context.
   This is often superior to UCB in non-stationary settings because it
   naturally balances exploration and exploitation based on uncertainty.
3. Non-stationarity handling: Instead of a gradual forgetting factor (which
   can lead to slow adaptation or numerical issues), use a "hard reset"
   strategy. If the reward prediction error is consistently high (indicating
   a regime change), reset the model to a prior state (identity matrix).
   This allows for rapid adaptation to new regimes.
4. Adaptive reset threshold: Monitor the normalized prediction error. If it
   exceeds a threshold, trigger a reset. The threshold can be adaptive based
   on the recent variance of rewards.
"""

import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Regularization
        
        # Global model parameters
        self.A = np.eye(dim) * self.lam
        self.b = np.zeros(dim)
        
        # Tracking for non-stationarity detection
        self.t = 0
        self.prediction_errors = []
        self.reward_history = []
        self.error_window = 50
        self.reward_window = 50
        
        # Reset parameters
        self.reset_threshold = 2.5  # Standard deviations
        self.min_resets = 100       # Minimum steps before reset
        self.last_reset = 0
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Precompute Cholesky for initial state
        self.L = np.linalg.cholesky(self.A)
        self.L_valid = True

    def _sample_theta(self):
        """Sample theta from N(theta_hat, A^{-1}) using Cholesky."""
        if not self.L_valid:
            try:
                self.L = np.linalg.cholesky(self.A)
                self.L_valid = True
            except np.linalg.LinAlgError:
                # Fallback: add jitter and retry
                self.A += np.eye(self.d) * 1e-6
                try:
                    self.L = np.linalg.cholesky(self.A)
                    self.L_valid = True
                except np.linalg.LinAlgError:
                    return np.zeros(self.d)
        
        try:
            theta_hat = np.linalg.solve(self.A, self.b)
            # Sample: theta = theta_hat + L^{-T} z, where z ~ N(0, I)
            # Note: A = L L^T, so A^{-1} = L^{-T} L^{-1}
            # We want sqrt(A^{-1}) ~ L^{-T}
            z = np.random.randn(self.d)
            # Solve L^T v = z => v = L^{-T} z
            v = np.linalg.solve(self.L.T, z)
            return theta_hat + v
        except np.linalg.LinAlgError:
            self.L_valid = False
            # Fallback to direct solve if Cholesky fails
            try:
                theta_hat = np.linalg.solve(self.A, self.b)
                cov = np.linalg.inv(self.A)
                return np.random.multivariate_normal(theta_hat, cov)
            except:
                return np.zeros(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fast fallback: greedy based on current mean
            try:
                theta_hat = np.linalg.solve(self.A, self.b)
                scores = [theta_hat @ context] * self.n
                return int(np.argmax(scores))
            except:
                return int(np.random.randint(self.n))

        # Thompson Sampling: sample one theta and evaluate all arms
        theta = self._sample_theta()
        scores = [theta @ context] * self.n
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for minimal update
        if time.time() > self.deadline:
            self.A += np.outer(context, context)
            self.b += reward * context
            self.L_valid = False
            return

        # Compute prediction error for non-stationarity detection
        try:
            theta_hat = np.linalg.solve(self.A, self.b)
            pred = theta_hat @ context
            error = reward - pred
        except:
            error = 0.0

        # Track errors and rewards
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.error_window:
            self.prediction_errors.pop(0)
        
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)

        # Check for regime change
        if self.t > self.min_resets and len(self.prediction_errors) >= 20:
            recent_errors = self.prediction_errors[-20:]
            mean_err = np.mean(recent_errors)
            std_err = np.std(recent_errors)
            
            # Normalize current error by recent std
            if std_err > 1e-6:
                z_score = abs(error - mean_err) / std_err
            else:
                z_score = 0.0

            # Adaptive threshold: lower threshold if reward variance is low
            reward_std = np.std(self.reward_history) if len(self.reward_history) > 5 else 1.0
            adaptive_threshold = self.reset_threshold * (1.0 + 0.5 * reward_std)
            
            if z_score > adaptive_threshold:
                # Trigger hard reset
                self.A = np.eye(self.d) * self.lam
                self.b = np.zeros(self.d)
                self.L = np.linalg.cholesky(self.A)
                self.L_valid = True
                self.prediction_errors = []
                self.reward_history = []
                self.last_reset = self.t
                # Continue to update with current observation after reset
                self.A += np.outer(context, context)
                self.b += reward * context
                self.L_valid = False
                return

        # Standard update
        self.A += np.outer(context, context)
        self.b += reward * context
        self.L_valid = False
