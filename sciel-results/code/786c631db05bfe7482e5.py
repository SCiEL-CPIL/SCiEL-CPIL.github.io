# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit with adaptive per-arm LinUCB
    and EWMA fallback. Uses exponentially weighted regression with adaptive
    forgetting factor tuned to observed reward variance.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Per-arm LinUCB with adaptive forgetting
        self.gamma = 0.97
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA as fallback
        self.ewma_alpha = 0.08
        self.ewma_values = np.zeros(n_arms)
        self.ewma_counts = np.zeros(n_arms)
        
        # Exploration
        self.base_beta = 1.2
        self.current_beta = self.base_beta
        
        # Tracking
        self.recent_rewards = []
        self.recent_window = 30
        self.best_arm = 0
        self.arm_recent = {a: [] for a in range(n_arms)}
        self.arm_history_len = 40
        self.reward_std = 1.0
        self.reward_mean = 0.0

    def _within_deadline(self):
        return time.time() < self.deadline

    def _compute_scores(self, context):
        """Compute UCB scores for all arms using per-arm LinUCB."""
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Solve A*theta = b
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
            mean = np.dot(theta, context)
            # Variance: context^T A^{-1} context
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(self.A[a])
                # Solve L y = context, then var = y^T y
                y = np.linalg.solve(L, context)
                var = np.dot(y, y)
            except np.linalg.LinAlgError:
                var = 1.0
            scores[a] = mean + self.current_beta * np.sqrt(var)
        return scores

    def select(self, context):
        self.t += 1
        
        # Deadline guard
        if not self._within_deadline():
            return self.best_arm
        
        # Adaptive beta: decay exploration over time, but adapt to reward scale
        progress = min(self.t / 500.0, 1.0)
        self.current_beta = self.base_beta * (1.0 - 0.6 * progress)
        
        if self.t <= 5:
            # Pure exploration in first few rounds
            return int(np.random.randint(self.n))
        
        scores = self._compute_scores(context)
        
        # Blend with EWMA if LinUCB is uncertain (high variance)
        # If t is small, EWMA has less data, so rely more on UCB
        if self.t > 20:
            ewma_confidence = min(self.t / 100.0, 1.0)
            # Normalize scores to comparable scale
            score_range = np.max(scores) - np.min(scores)
            if score_range > 1e-8:
                normalized_ucb = (scores - np.min(scores)) / score_range
                normalized_ewma = (self.ewma_values - np.min(self.ewma_values)) / (np.max(self.ewma_values) - np.min(self.ewma_values) + 1e-8)
                blend = 0.3 * ewma_confidence
                scores = (1.0 - blend) * scores + blend * (np.min(scores) + normalized_ewma * score_range)
        
        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        # Deadline guard
        if not self._within_deadline():
            return
        
        # Update per-arm LinUCB with adaptive forgetting
        outer = np.outer(context, context)
        self.A[arm] = self.gamma * self.A[arm] + outer
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update EWMA
        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1
        
        # Track recent rewards for adaptive forgetting
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_window:
            self.recent_rewards.pop(0)
        
        # Track arm reward history for best_arm
        self.arm_recent[arm].append(reward)
        if len(self.arm_recent[arm]) > self.arm_history_len:
            self.arm_recent[arm].pop(0)
        
        # Update best arm based on recent history
        arm_avgs = {}
        for a in range(self.n):
            if self.arm_recent[a]:
                arm_avgs[a] = np.mean(self.arm_recent[a])
            else:
                arm_avgs[a] = -np.inf
        if arm_avgs:
            self.best_arm = max(arm_avgs, key=arm_avgs.get)
        
        # Adaptive forgetting: adjust gamma based on reward variance
        if len(self.recent_rewards) >= 10:
            r_std = np.std(self.recent_rewards)
            r_mean = np.mean(self.recent_rewards)
            # Higher variance suggests more non-stationarity -> faster forgetting
            # Target gamma in [0.93, 0.99]
            target_gamma = 0.99 - 0.06 * min(r_std / (abs(r_mean) + 0.1), 1.0)
            target_gamma = max(0.93, min(0.99, target_gamma))
            # Smoothly adjust
            self.gamma = 0.95 * self.gamma + 0.05 * target_gamma
# EVOLVE-BLOCK-END
