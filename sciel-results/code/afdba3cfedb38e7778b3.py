# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Decaying Linear Bandit with exponential forgetting for non-stationary rewards."""

    def __init__(self, n_arms, dim, gamma=0.98, ridge=1e-2, beta0=1.5):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension for intercept
        self.gamma = gamma
        self.ridge = ridge
        self.beta0 = beta0
        
        # Per-arm sufficient statistics
        # A: (D, D) decayed sum of x x^T
        # b: (D,) decayed sum of x y
        # We initialize A with ridge * I to ensure invertibility and regularization
        self.A = [np.eye(self.D) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(self.D) for _ in range(n_arms)]
        
        # Track number of updates per arm for potential initial exploration boost
        self.count = [0 for _ in range(n_arms)]
        
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
        
        # Decay exploration slightly over time, but keep it significant
        # Using a constant beta is often robust, but slight decay helps if noise is constant
        beta = self.beta0
        
        for a in range(self.n):
            # Compute theta = A^{-1} b
            # Add ridge regularization to ensure stability if A becomes ill-conditioned
            A_reg = self.A[a] + self.ridge * np.eye(self.D)
            
            try:
                # Use solve instead of inv for numerical stability and speed
                # theta = A_reg^{-1} b
                theta = np.linalg.solve(A_reg, self.b[a])
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular (rare with ridge)
                theta = np.linalg.pinv(A_reg) @ self.b[a]
                
            mu = theta @ x_aug
            
            # Bonus: beta * sqrt(x^T A_reg^{-1} x)
            # We need A_reg^{-1} x_aug
            try:
                A_inv_x = np.linalg.solve(A_reg, x_aug)
            except np.linalg.LinAlgError:
                A_inv_x = np.linalg.pinv(A_reg) @ x_aug
                
            v = x_aug @ A_inv_x
            bonus = beta * np.sqrt(max(v, 0.0))
            
            scores[a] = mu + bonus
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Apply exponential decay to existing statistics
        self.A[arm] *= self.gamma
        self.b[arm] *= self.gamma
        
        # Add new observation
        self.A[arm] += np.outer(x_aug, x_aug)
        self.b[arm] += x_aug * reward
        
        self.count[arm] += 1
        self.t += 1
# EVOLVE-BLOCK-END
