# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Exponentially-weighted linear bandit with Sherman-Morrison rank-1 updates and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as identity (ridge) for each arm - stronger ridge for stability
        self.A = [np.eye(dim) * 0.2 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Precompute inverses for Sherman-Morrison updates
        self.A_inv = [np.eye(dim) / 0.2 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting factor - start aggressive for fast non-stationary adaptation
        self.gamma = 0.90
        # Exploration parameter - slightly larger for better initial exploration
        self.alpha = 1.8
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s
        self.round_count = 0
        # Per-arm visit counts and prediction error tracking
        self.Visits = [0] * n_arms
        self.pred_errors = [np.zeros(0) for _ in range(n_arms)]
        self.recent_errors = [[] for _ in range(n_arms)]

    def _sherman_morrison_update(self, arm, context, reward):
        """Update A_inv and theta using Sherman-Morrison formula after rank-1 update."""
        x = context
        inv = self.A_inv[arm]
        g = self.gamma
        x_over_g = x / g
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse
            self.A[arm] = g * self.A[arm] + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        # A_new^{-1} = (1/g) * (inv - np.outer(Ax, Ax) / denom)
        new_inv = (inv - np.outer(Ax, Ax) / denom) / g
        self.A_inv[arm] = new_inv
        
        # Update b: b_new = g * b + reward * x
        self.b[arm] = g * self.b[arm] + reward * x
        # Update theta = A_new^{-1} @ b_new
        self.theta[arm] = new_inv @ self.b[arm]
        
        # Also update A for potential fallback use
        self.A[arm] = g * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return a deterministic arm based on context hash to avoid exceptions
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.A_inv[a]
            mean = theta @ context
            # UCB bonus with adaptive alpha
            var = context @ inv @ context
            # Scale bonus by sqrt of number of visits to this arm for better calibration
            # Tighter visit-based scaling to reduce over-exploration of well-visited arms
            visit_bonus_scale = 1.0 / (1.0 + 0.15 * self.Visits[a])
            bonus = self.alpha * visit_bonus_scale * np.sqrt(max(var, 0.0))
            scores.append(mean + bonus)
        
        scores = np.array(scores)
        
        # Epsilon-greedy: with small probability, explore randomly
        if self.round_count < 100 and np.random.random() < 0.1:
            return int(np.random.randint(self.n))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Track prediction error
        pred = self.theta[arm] @ context
        pred_err = abs(reward - pred)
        
        # Update per-arm error tracking (keep last 20 errors)
        self.recent_errors[arm].append(pred_err)
        if len(self.recent_errors[arm]) > 20:
            self.recent_errors[arm].pop(0)
        
        # Adaptive forgetting: more responsive to prediction errors
        # Use recent error average for better signal
        if self.recent_errors[arm]:
            avg_err = np.mean(self.recent_errors[arm])
        else:
            avg_err = pred_err
        
        # Target gamma: high error -> low gamma (faster forgetting)
        # Map error to gamma range [0.85, 0.99] - wider range for more aggressive adaptation
        target_gamma = 0.99 - 0.14 * np.tanh(avg_err / 1.2)
        # Faster adaptation: 0.20 weight for quicker response to drift
        self.gamma = 0.80 * self.gamma + 0.20 * target_gamma
        self.gamma = np.clip(self.gamma, 0.85, 0.99)
        
        # Perform the update
        self._sherman_morrison_update(arm, context, reward)
        
        # Track visits
        self.Visits[arm] += 1
        
        # Adjust exploration: reduce alpha over time but keep some exploration
        # Slightly faster decay to exploit more after initial phase
        self.alpha = max(0.6, self.alpha * 0.99990)
# EVOLVE-BLOCK-END
