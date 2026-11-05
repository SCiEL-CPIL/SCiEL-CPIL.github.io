# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit with intercept term, adaptive forgetting, and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim + 1  # Augmented dimension with intercept
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 110s deadline (10s safety margin)
        
        # Inverse covariance matrices for each arm (augmented)
        self.V = [np.eye(self.d) for _ in range(n_arms)]
        # Reward-weighted feature vectors (augmented)
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.alpha = 1.5  # Exploration parameter
        self.gamma = 0.98  # Forgetting factor
        self.total_rounds = 0
        self.reward_history = [0.0 for _ in range(n_arms)]

    def _augment(self, context):
        """Augment context with intercept term."""
        context = np.asarray(context, dtype=np.float64)
        return np.append(context, 1.0)

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() > self.deadline

    def _select_safe_arm(self):
        """Return a safe arm when deadline is approaching."""
        # Return the arm with highest average reward history
        if any(r != 0 for r in self.reward_history):
            return int(np.argmax(self.reward_history))
        return 0

    def select(self, context):
        # Deadline guard
        if self._check_deadline():
            return self._select_safe_arm()
        
        x = self._augment(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Compute theta = V @ b
            theta = self.V[a] @ self.b[a]
            
            # Predicted reward
            pred = theta @ x
            
            # Uncertainty bonus: sqrt(x^T V x)
            Vx = self.V[a] @ x
            uncertainty = np.sqrt(max(0.0, x @ Vx))
            
            scores[a] = pred + self.alpha * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if self._check_deadline():
            return
        
        x = self._augment(context)
        self.total_rounds += 1
        
        # Update reward history for adaptive exploration
        self.reward_history[arm] = 0.95 * self.reward_history[arm] + 0.05 * reward
        
        # Adaptive forgetting: decrease gamma when rewards are volatile
        if self.total_rounds > 10:
            vol = np.std(self.reward_history)
            target_gamma = 0.995 - 0.01 * min(vol, 1.0)
            self.gamma = 0.9 * self.gamma + 0.1 * target_gamma
        
        # Update with forgetting using Sherman-Morrison-like approach
        # V_new = gamma * V_old + outer(x, x)
        # We need to maintain V as the inverse of the Gram matrix
        # Instead, we'll track the Gram matrix and invert periodically, 
        # or use the matrix inversion lemma
        
        # Actually, let's track A (Gram matrix) and b, and compute V = A^{-1} when needed
        # For efficiency, let's store A and b, and compute V lazily
        
        # Update A and b with forgetting
        # We need to store A as well
        if not hasattr(self, 'A'):
            self.A = [np.eye(self.d) * 0.1 for _ in range(self.n)]
        
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + reward * x
        self.count[arm] += 1
        
        # Update V = A^{-1}
        try:
            self.V[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.V[arm] = np.eye(self.d)
        
        # Adaptive exploration: decrease alpha over time but keep some exploration
        self.alpha = max(0.5, 1.5 * np.exp(-self.total_rounds / 300.0))
# EVOLVE-BLOCK-END
