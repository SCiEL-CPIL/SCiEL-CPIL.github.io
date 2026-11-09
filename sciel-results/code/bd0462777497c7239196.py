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
        # Precompute inverse matrices for faster scoring
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        
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
            # theta = A^{-1} b
            theta = self.A_inv[a] @ self.b[a]
            mean = np.dot(theta, context)
            # Variance: context^T A^{-1} context
            var = context @ self.A_inv[a] @ context
            scores[a] = mean + self.current_beta * np.sqrt(max(var, 0.0))
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
        
        # Update inverse using Sherman-Morrison-Woodbury
        # A_new = gamma * A_old + x x^T
        # A_new^{-1} = (1/gamma) * A_old^{-1} - (1/gamma^2) * A_old^{-1} x x^T A_old^{-1} / (1/gamma + x^T A_old^{-1} x)
        # Actually: A_new = gamma * A_old + x x^T
        # Let A_old^{-1} = V. Then A_new^{-1} = (1/gamma) V - (1/gamma^2) V x x^T V / (1/gamma + x^T V x)
        V = self.A_inv[arm]
        Vx = V @ context
        denom = 1.0 / self.gamma + context @ Vx
        if abs(denom) > 1e-12:
            self.A_inv[arm] = (1.0 / self.gamma) * V - (1.0 / (self.gamma * self.gamma)) * np.outer(Vx, Vx) / denom
        else:
            # Fallback to direct inverse
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d)
        
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
