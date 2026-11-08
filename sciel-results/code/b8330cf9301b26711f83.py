# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit with adaptive per-arm LinUCB
    and variance-adaptive Thompson Sampling.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin
        
        # Per-arm linear model parameters with exponential forgetting
        self.gamma = 0.99
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA of arm rewards for quick tracking
        self.ewma_alpha = 0.08
        self.ewma_values = np.zeros(n_arms)
        self.ewma_counts = np.zeros(n_arms)
        
        # Adaptive parameters
        self.beta = 1.0
        self.best_arm = 0
        self.recent_rewards = []
        self.recent_window = 30
        
        # Variance tracking for adaptive forgetting
        self.reward_variances = np.zeros(n_arms)
        self.reward_means = np.zeros(n_arms)
        self.var_window = 20
        self.reward_history = {a: [] for a in range(n_arms)}
        
        # Precompute inverses for speed
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]

    def _within_deadline(self):
        return time.time() < self.deadline

    def _update_inverse(self, arm):
        """Update inverse of A using Sherman-Morrison formula for rank-1 update."""
        A = self.A[arm]
        inv_A = self.inv_A[arm]
        # After forgetting: A_new = gamma * A_old + x x^T
        # We track A and compute inv_A periodically or use Sherman-Morrison
        # For simplicity and numerical stability, recompute when needed
        try:
            self.inv_A[arm] = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.inv_A[arm] = np.eye(self.d)

    def _compute_theta(self, arm):
        """Compute theta = A^{-1} b for arm."""
        try:
            return self.inv_A[arm] @ self.b[arm]
        except Exception:
            return np.zeros(self.d)

    def select(self, context):
        self.t += 1
        
        # Deadline guard
        if not self._within_deadline():
            return self.best_arm
        
        # Adaptive beta: decrease exploration over time
        self.beta = 1.0 / (1.0 + 0.0005 * self.t)
        
        # Thompson Sampling: sample from posterior for each arm
        samples = np.zeros(self.n)
        for a in range(self.n):
            theta = self._compute_theta(a)
            mean = theta @ context
            inv_A = self.inv_A[a]
            # Variance of the prediction
            var = context @ inv_A @ context
            # Sample from N(mean, var)
            samples[a] = np.random.normal(mean, np.sqrt(max(var, 1e-10)))
        
        # Blend with EWMA for robustness
        if self.t > 10:
            # Weight Thompson samples more as we gather data
            ts_weight = min(0.8, 0.3 + 0.01 * self.t)
            ew_weight = 1.0 - ts_weight
            # Normalize EWMA to comparable scale
            ewma_norm = self.ewma_values - np.min(self.ewma_values)
            if np.max(ewma_norm) > 1e-10:
                ewma_norm = ewma_norm / np.max(ewma_norm)
            else:
                ewma_norm = np.ones(self.n) / self.n
            samples = ts_weight * samples + ew_weight * ewma_norm
        
        arm = int(np.argmax(samples))
        return arm

    def update(self, context, arm, reward):
        self.t += 1
        
        if not self._within_deadline():
            return
        
        # Update per-arm linear model with forgetting
        outer = np.outer(context, context)
        self.A[arm] = self.gamma * self.A[arm] + outer
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update inverse (do this every few steps to save computation)
        if self.t % 5 == 0:
            self._update_inverse(arm)
        
        # Update EWMA
        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1
        
        # Track reward history for variance estimation
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.var_window:
            self.reward_history[arm].pop(0)
        
        # Adaptive forgetting: increase gamma (slower forgetting) if stable, decrease if volatile
        if len(self.reward_history[arm]) >= 5:
            recent = np.array(self.reward_history[arm])
            reward_var = np.var(recent)
            reward_mean = np.mean(recent)
            # If variance is high, we want faster forgetting (lower gamma)
            # If variance is low, we can afford slower forgetting (higher gamma)
            target_gamma = 0.995 - 0.02 * min(reward_var, 1.0)
            # Smoothly adjust gamma
            self.gamma = 0.9 * self.gamma + 0.1 * target_gamma
            self.gamma = np.clip(self.gamma, 0.97, 0.998)
        
        # Track recent rewards for best arm selection
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_window:
            self.recent_rewards.pop(0)
        
        # Update best arm based on EWMA
        if self.t > 5:
            # Only consider arms that have been tried at least once
            valid_arms = [a for a in range(self.n) if self.ewma_counts[a] > 0]
            if valid_arms:
                self.best_arm = max(valid_arms, key=lambda a: self.ewma_values[a])
# EVOLVE-BLOCK-END
