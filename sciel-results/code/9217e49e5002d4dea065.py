import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Forgetting factors - adaptive per arm
        self.gamma_global = 0.99
        self.gamma_arm_base = 0.85
        self.gamma_arm = np.full(n_arms, self.gamma_arm_base)
        
        # Global Linear Model
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)
        self.inv_A_global = np.eye(dim)
        
        # Per-Arm Linear Models
        self.A_arm = [np.eye(dim) * 0.5 for _ in range(n_arms)]
        self.b_arm = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A_arm = [np.eye(dim) * 2.0 for _ in range(n_arms)]
        
        # UCB parameters
        self.base_beta = 1.0
        
        # Drift detection: track recent reward changes per arm
        self.ewma_alpha = 0.15
        self.ewma_values = np.zeros(n_arms)
        self.prev_ewma = np.zeros(n_arms)
        self.drift_signal = np.zeros(n_arms)
        
        # Track arm pulls for adaptive gamma
        self.arm_pulls = np.zeros(n_arms)
        self.last_reward = np.zeros(n_arms)
        self.reward_diff_sum = np.zeros(n_arms)
        self.reward_diff_count = np.zeros(n_arms)
        
        # Best arm fallback
        self.best_arm = 0
        self.ewma_counts = np.zeros(n_arms)

    def _within_deadline(self):
        return time.time() < self.deadline

    def _sm_update_inverse(self, A_inv, x, gamma):
        v = A_inv @ x
        denom = gamma + x @ v
        if abs(denom) < 1e-10:
            denom = 1e-10
        sm_term = np.outer(v, v) / denom
        M_inv = A_inv - sm_term
        new_inv = M_inv / gamma
        new_inv = (new_inv + new_inv.T) * 0.5
        return new_inv

    def _adapt_gamma(self, arm, reward):
        """Adaptively adjust forgetting rate based on observed drift."""
        self.arm_pulls[arm] += 1
        
        if self.arm_pulls[arm] > 5:
            diff = abs(reward - self.last_reward[arm])
            self.reward_diff_sum[arm] = 0.9 * self.reward_diff_sum[arm] + 0.1 * diff
            self.reward_diff_count[arm] += 1
            
            # If recent changes are large, decrease gamma (faster forgetting)
            if self.reward_diff_sum[arm] > 0.3:
                self.gamma_arm[arm] = max(0.7, self.gamma_arm[arm] * 0.98)
            elif self.reward_diff_sum[arm] < 0.05:
                self.gamma_arm[arm] = min(0.95, self.gamma_arm[arm] * 1.02)
        
        self.last_reward[arm] = reward

    def select(self, context):
        self.t += 1
        
        if not self._within_deadline():
            return self.best_arm
            
        context = np.asarray(context)
        scores = np.zeros(self.n)
        
        # Compute global theta once
        theta_global = self.inv_A_global @ self.b_global
        
        # Time-decaying exploration: start higher, decay over time
        time_decay = np.exp(-self.t / 500.0)
        time_bonus = 0.3 * time_decay
        
        for a in range(self.n):
            theta_arm = self.inv_A_arm[a] @ self.b_arm[a]
            theta = theta_global + theta_arm
            
            mean = theta @ context
            
            # Variance: combine global and arm uncertainty
            v_global = context @ self.inv_A_global
            v_arm = context @ self.inv_A_arm[a]
            var = context @ (v_global + v_arm)
            
            if var < 1e-10:
                var = 1e-10
            
            # Adaptive beta: boost exploration when drift detected
            drift_boost = 1.0 + 0.5 * self.drift_signal[a]
            beta = self.base_beta * drift_boost + time_bonus
            
            scores[a] = mean + beta * np.sqrt(var)
        
        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        self.t += 1
        
        if not self._within_deadline():
            return
            
        context = np.asarray(context)
        
        # Update global model
        self.inv_A_global = self._sm_update_inverse(self.inv_A_global, context, self.gamma_global)
        self.b_global = self.gamma_global * self.b_global + reward * context
        
        # Update arm model
        g = self.gamma_arm[arm]
        self.inv_A_arm[arm] = self._sm_update_inverse(self.inv_A_arm[arm], context, g)
        self.b_arm[arm] = g * self.b_arm[arm] + reward * context
        
        # Update drift signal
        self.prev_ewma[arm] = self.ewma_values[arm]
        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1
        
        # Drift detection: compare current EWMA to previous
        if self.ewma_counts[arm] > 10:
            change = abs(self.ewma_values[arm] - self.prev_ewma[arm])
            self.drift_signal[arm] = 0.9 * self.drift_signal[arm] + 0.1 * min(1.0, change * 5)
        
        # Adapt gamma based on observed reward variability
        self._adapt_gamma(arm, reward)
        
        # Update best arm for fallback
        if self.t % 5 == 0:
            valid = [a for a in range(self.n) if self.ewma_counts[a] > 0]
            if valid:
                self.best_arm = max(valid, key=lambda a: self.ewma_values[a])
