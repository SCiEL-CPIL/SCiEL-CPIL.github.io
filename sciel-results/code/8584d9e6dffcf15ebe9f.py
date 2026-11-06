"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Uses per-arm ridge regression with exponential forgetting to track drifting
    arm weights. Adaptive decay responds to observed reward drift. UCB-style
    exploration bonus with periodic numerical stabilization.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Base forgetting factor
        self.base_decay = 0.95
        # Adaptive decay: adjust based on drift detection
        self.decay = self.base_decay
        
        # Exploration parameter - UCB style
        self.alpha = 1.0
        
        # Round counter
        self.round_count = 0
        
        # Drift detection: track recent rewards per arm
        self.recent_rewards = [deque(maxlen=30) for _ in range(n_arms)]
        
        # Numerical stabilization: recompute full inverse periodically
        self.recompute_interval = 50
        self.update_count = [0] * n_arms
        
        # Deadline guard
        self.t0 = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        return (time.time() - self.t0) > self.deadline

    def _estimate_drift(self, arm):
        """Estimate how much the arm's reward pattern is drifting."""
        if len(self.recent_rewards[arm]) < 10:
            return 0.0
        rewards = list(self.recent_rewards[arm])
        mid = len(rewards) // 2
        first_half = np.mean(rewards[:mid])
        second_half = np.mean(rewards[mid:])
        overall_std = np.std(rewards) + 1e-6
        drift = abs(second_half - first_half) / overall_std
        return min(drift, 2.0)

    def _adapt_decay(self, arm):
        """Adjust decay factor based on detected drift."""
        drift = self._estimate_drift(arm)
        self.decay = max(0.85, min(0.97, 0.97 - 0.045 * drift))

    def select(self, context):
        if self._check_deadline():
            scores = np.array([self.theta[a] @ context for a in range(self.n)])
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64).flatten()
        scores = np.zeros(self.n)
        
        if self.round_count < 20:
            alpha = 2.0
        elif self.round_count < 100:
            alpha = 1.0 + 1.0 * np.exp(-(self.round_count - 20) / 80.0)
        else:
            alpha = max(0.3, 1.0 * np.exp(-(self.round_count - 100) / 200.0))
        
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.inv_A[a]
            mean_val = theta @ context
            variance = context @ inv @ context
            variance = max(0.0, variance)
            std_val = np.sqrt(variance)
            scores[a] = mean_val + alpha * std_val
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        arm = int(arm)
        reward = float(reward)
        
        self.recent_rewards[arm].append(reward)
        self._adapt_decay(arm)
        
        decay = self.decay
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
        
        d = self.d
        self.update_count[arm] += 1
        
        if self.update_count[arm] % self.recompute_interval == 0:
            try:
                self.inv_A[arm] = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(d))
                self.theta[arm] = self.inv_A[arm] @ self.b[arm]
            except np.linalg.LinAlgError:
                self.A[arm] += 1e-4 * np.eye(d)
                try:
                    self.inv_A[arm] = np.linalg.inv(self.A[arm])
                    self.theta[arm] = self.inv_A[arm] @ self.b[arm]
                except np.linalg.LinAlgError:
                    pass
        else:
            try:
                inv_old = self.inv_A[arm]
                B_inv = (1.0 / decay) * inv_old
                x = context
                xB = B_inv @ x
                denom = 1.0 + x @ xB
                if abs(denom) > 1e-12:
                    inv_new = B_inv - np.outer(xB, xB) / denom
                else:
                    inv_new = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(d))
                self.inv_A[arm] = inv_new
                self.theta[arm] = inv_new @ self.b[arm]
            except (np.linalg.LinAlgError, Exception):
                try:
                    self.inv_A[arm] = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(d))
                    self.theta[arm] = self.inv_A[arm] @ self.b[arm]
                except np.linalg.LinAlgError:
                    pass

        self.round_count += 1

# EVOLVE-BLOCK-END
