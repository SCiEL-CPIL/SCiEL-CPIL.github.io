# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    Uses per-arm linear models with a forgetting factor (discounted least squares)
    and UCB exploration. Adds variance-based exploration bonus and numerical stability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.alpha = 0.995  # forgetting factor
        self.ridge = 1e-3   # ridge regularization for stability

        # Per-arm sufficient statistics
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        
        # Track reward variance per arm for adaptive exploration
        self.reward_sum = [0.0 for _ in range(n_arms)]
        self.reward_sq_sum = [0.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        # Deadline guard: if close to timeout, use greedy based on current estimates
        if not self._check_deadline():
            # Fallback: return arm with highest mean estimate
            scores = []
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            # Compute theta using pre-computed inverse for speed
            theta = self.A_inv[a] @ self.b[a]
            mean_reward = theta @ context
            
            # Compute uncertainty using pre-computed inverse
            # var = context^T A_inv context
            var = context @ self.A_inv[a] @ context
            
            # Base UCB exploration
            exploration = 1.0 * np.sqrt(var)
            
            # Variance-based exploration bonus: increase exploration if recent rewards are variable
            if self.count[a] > 1:
                mean_r = self.reward_sum[a] / self.count[a]
                var_r = max(0.0, self.reward_sq_sum[a] / self.count[a] - mean_r**2)
                # Scale bonus by recent reward variance, decayed
                var_bonus = 0.5 * np.sqrt(var_r)
            else:
                var_bonus = 0.5
            
            score = mean_reward + exploration + var_bonus
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Update tracking statistics
        self.reward_sum[arm] = self.alpha * self.reward_sum[arm] + reward
        self.reward_sq_sum[arm] = self.alpha * self.reward_sq_sum[arm] + reward**2
        self.count[arm] = self.alpha * self.count[arm] + 1
        
        # Update sufficient statistics with forgetting factor
        outer = np.outer(context, context)
        self.A[arm] = self.alpha * self.A[arm] + outer
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
        # Update inverse using Sherman-Morrison or direct recompute
        # For small dim, direct recompute is fine and more stable
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # If singular, add more ridge
            self.A[arm] += 0.01 * np.eye(self.d)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
# EVOLVE-BLOCK-END
