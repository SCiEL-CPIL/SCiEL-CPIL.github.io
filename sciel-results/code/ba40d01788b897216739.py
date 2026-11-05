"""Non-stationary linear contextual bandit with adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive-forgetting LinUCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_decay = 0.99       # Mild forgetting each step
        self.shift_decay = 0.5       # Strong forgetting on regime shift
        self.shift_threshold = 2.0   # Residual/uncertainty threshold
        self.min_decay = 0.3         # Floor for decay factor
        self.max_decay = 0.999       # Ceiling for decay factor
        
        # UCB exploration parameter
        self.alpha = 1.0
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

    def _get_theta_and_var(self, arm, context):
        """Compute mean and variance for an arm given context."""
        # Add small regularization for numerical stability
        A = self.A[arm] + 1e-6 * np.eye(self.d)
        
        # Solve for theta: A theta = b
        try:
            theta = np.linalg.solve(A, self.b[arm])
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ self.b[arm]
            
        mu = theta @ context
        
        # Variance: context^T A^{-1} context
        # We can solve A v = context to get v = A^{-1} context, then var = context^T v
        try:
            v = np.linalg.solve(A, context)
            var = context @ v
        except np.linalg.LinAlgError:
            var = context @ (np.linalg.pinv(A) @ context)
            
        return mu, max(var, 1e-12)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                mu, var = self._get_theta_and_var(a, context)
                score = mu + self.alpha * np.sqrt(var)
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        scores = []
        for a in range(self.n):
            mu, var = self._get_theta_and_var(a, context)
            score = mu + self.alpha * np.sqrt(var)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Calculate prediction error to adapt decay
        # Use current model to predict
        mu, var = self._get_theta_and_var(arm, context)
        residual = reward - mu
        abs_residual = np.abs(residual)
        
        # Determine decay factor
        # If residual is large, assume regime change and forget faster
        # If residual is small, assume stable and keep memory
        if abs_residual > self.shift_threshold:
            decay = self.shift_decay
        else:
            # Interpolate or use base decay
            # A simple heuristic: if residual is very small, use max_decay
            # Otherwise base_decay
            if abs_residual < 0.5:
                decay = self.max_decay
            else:
                decay = self.base_decay

        # Update model with decay
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
