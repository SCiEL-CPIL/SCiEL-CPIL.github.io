# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Exponentially weighted linear bandit with recency bias for non-stationarity."""

    def __init__(self, n_arms, dim, alpha=0.95, ridge=1e-2, beta0=1.5):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension (intercept + context)
        self.alpha = alpha
        self.ridge = ridge
        self.beta0 = beta0
        
        # Per-arm statistics
        # A: (D, D) weighted sum of x x^T
        # b: (D,) weighted sum of x y
        self.A = [np.eye(self.D) * ridge for _ in range(n_arms)]
        self.b = [np.zeros(self.D) for _ in range(n_arms)]
        
        # Track total weight for normalization if needed, though usually A is sufficient for UCB
        # We update A and b by decaying existing values and adding new ones.
        
        self.t = 0

    def _augment(self, context):
        context = np.asarray(context, dtype=float).ravel()
        x_aug = np.zeros(self.D)
        x_aug[0] = 1.0
        x_aug[1:] = context
        return x_aug

    def select(self, context):
        x_aug = self._augment(context)
        scores = np.zeros(self.n)
        
        # Adaptive beta: start high, decay slowly
        beta = self.beta0 * (1.0 / (1.0 + 0.005 * self.t))
        
        for a in range(self.n):
            # A is already regularized (initialized with ridge * I and updated)
            # We need A_inv. For small D, inverse is fast.
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[a])
            
            theta = A_inv @ self.b[a]
            mu = theta @ x_aug
            
            # UCB bonus
            # v = x^T A_inv x
            v = x_aug @ A_inv @ x_aug
            bonus = beta * np.sqrt(max(v, 0.0))
            
            scores[a] = mu + bonus
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Decay existing statistics
        self.A[arm] *= self.alpha
        self.b[arm] *= self.alpha
        
        # Add new observation
        self.A[arm] += np.outer(x_aug, x_aug)
        self.b[arm] += x_aug * reward
        
        self.t += 1
# EVOLVE-BLOCK-END
