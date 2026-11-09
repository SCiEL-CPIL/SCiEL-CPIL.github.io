# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary Linear Contextual Bandit with Sliding Window LinUCB.
    
    Uses an exact sliding window least squares estimator for each arm.
    This avoids the "memory tail" of exponential forgetting, allowing for
    faster adaptation to regime shifts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Sliding window parameters
        self.window_size = 40
        # Store (context, reward) pairs for each arm
        self.windows = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Current A and b matrices for each arm, updated incrementally
        # A = sum(x x^T) + lambda I, b = sum(r x)
        self.reg_lambda = 0.1
        self.A = [np.eye(dim) * self.reg_lambda for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Exploration
        self.base_beta = 1.5
        self.current_beta = self.base_beta
        
        # Tracking for adaptive behavior
        self.best_arm = 0
        self.recent_rewards = []
        self.recent_window = 20
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Initialize with dummy data to avoid empty window issues in early steps?
        # No, handle empty window in compute.

    def _within_deadline(self):
        return time.time() < self.deadline

    def _update_arm_model(self, arm, context, reward):
        """
        Add new sample to the window and update A, b incrementally.
        If window is full, remove the oldest sample.
        """
        w = self.windows[arm]
        
        if len(w) == self.window_size:
            # Remove oldest
            old_ctx, old_rwd = w[0]
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_rwd * old_ctx
            
        # Add new
        w.append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def _compute_scores(self, context):
        """
        Compute UCB scores for all arms using the current windowed LS estimates.
        """
        scores = np.zeros(self.n)
        ctx_norm = np.linalg.norm(context)
        if ctx_norm < 1e-6:
            context = np.zeros(self.d)
        else:
            # Normalize context for stability? 
            # The LS model is scale invariant in theta if we adjust, 
            # but UCB variance depends on context magnitude.
            # Let's keep raw context to match the reward scale.
            pass

        for a in range(self.n):
            # Solve A theta = b
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                
            mean = np.dot(theta, context)
            
            # Variance: context^T A^{-1} context
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(self.A[a])
                y = np.linalg.solve(L, context)
                var = np.dot(y, y)
            except np.linalg.LinAlgError:
                # Fallback if A is not PD (should not happen with reg_lambda > 0)
                var = 1.0
                
            scores[a] = mean + self.current_beta * np.sqrt(var)
            
        return scores

    def select(self, context):
        self.t += 1
        
        # Deadline guard
        if not self._within_deadline():
            return self.best_arm
        
        # Adaptive beta: 
        # Start with high exploration, decay over time.
        # Also boost exploration if recent rewards are volatile (non-stationary signal)
        progress = min(self.t / 300.0, 1.0)
        base_decay = 1.0 - 0.5 * progress
        
        # Calculate volatility of recent rewards
        if len(self.recent_rewards) > 5:
            r_std = np.std(self.recent_rewards)
            r_mean = np.mean(self.recent_rewards)
            # If std is high relative to mean, we are in a volatile regime
            volatility_factor = 1.0 + min(r_std / (abs(r_mean) + 0.1), 1.0)
        else:
            volatility_factor = 1.5 # High initial exploration
        
        self.current_beta = self.base_beta * base_decay * volatility_factor

        # Pure exploration for first few rounds to populate windows
        if self.t <= 5:
            return int(np.random.randint(self.n))
        
        scores = self._compute_scores(context)
        
        # Simple argmax
        arm = int(np.argmax(scores))
        
        return arm

    def update(self, context, arm, reward):
        # Deadline guard
        if not self._within_deadline():
            return
        
        # Update the model for the selected arm
        self._update_arm_model(arm, context, reward)
        
        # Track recent rewards for adaptive beta
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_window:
            self.recent_rewards.pop(0)
        
        # Update best arm heuristic:
        # Use the mean of the last few rewards for each arm if available,
        # otherwise fall back to the UCB mean estimate.
        # For simplicity and speed, we just track the arm that gave the highest
        # recent reward in a small window.
        # To avoid O(n) list operations every step, we can just use the current
        # UCB means as a proxy for "best" in the deadline guard.
        # However, for the deadline guard, we want the arm with the highest
        # *recent* performance.
        
        # Let's maintain a simple EWMA of the best arm based on recent rewards
        # We don't have per-arm recent rewards stored efficiently in this new design
        # (we have windows, but accessing mean of window is O(W)).
        # W=40 is small. Let's just compute it if needed? 
        # No, let's just pick the arm with the highest UCB mean (without exploration)
        # as the "best_arm" for the deadline guard.
        
        # Actually, a better "best_arm" for non-stationary is the one with the
        # highest average reward in the last 10 steps.
        # We can approximate this by keeping a small deque of (arm, reward) for the last 10 steps.
        
        if not hasattr(self, 'recent_arm_rewards'):
            self.recent_arm_rewards = deque(maxlen=10)
            
        self.recent_arm_rewards.append((arm, reward))
        
        # Calculate average reward per arm in this small recent window
        arm_sums = {a: 0.0 for a in range(self.n)}
        arm_counts = {a: 0 for a in range(self.n)}
        
        for a, r in self.recent_arm_rewards:
            arm_sums[a] += r
            arm_counts[a] += 1
            
        # Find arm with highest average, tie-break by UCB mean
        best_val = -np.inf
        best_a = self.best_arm
        for a in range(self.n):
            if arm_counts[a] > 0:
                avg = arm_sums[a] / arm_counts[a]
                if avg > best_val:
                    best_val = avg
                    best_a = a
            else:
                # If no recent data, use UCB mean
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    mean = np.dot(theta, context) # Context is from select, but update doesn't have context? 
                    # Wait, update signature is update(context, arm, reward). So we have context.
                    # But this is inside update, so we have context.
                    # However, doing solve here is expensive. 
                    # Let's just keep the previous best_arm if no recent data.
                except:
                    pass
        
        self.best_arm = best_a

# EVOLVE-BLOCK-END
