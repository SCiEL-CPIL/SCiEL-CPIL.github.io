"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Linear UCB with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices A and mean vectors b
        # A is the precision matrix (inverse covariance)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor
        self.alpha = 0.99
        # Exploration parameter for UCB
        self.c = 3.0
        # Regularization to ensure A is positive definite
        self.reg = 1e-3

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Regularize A for stability
            A_reg = self.A[a] + self.reg * np.eye(self.d)
            
            # Compute theta_hat = A^-1 b
            try:
                theta_hat = np.linalg.solve(A_reg, self.b[a])
            except np.linalg.LinAlgError:
                theta_hat = np.linalg.lstsq(A_reg, self.b[a], rcond=None)[0]
            
            # Compute confidence bonus
            # We want sqrt(x^T A^-1 x)
            # Let A = L L^T. Then x^T A^-1 x = || L^-1 x ||^2
            try:
                L = np.linalg.cholesky(A_reg)
                # Solve L z = x
                z = np.linalg.solve(L, context)
                variance = np.dot(z, z)
            except np.linalg.LinAlgError:
                # Fallback to inverse if Cholesky fails
                try:
                    A_inv = np.linalg.inv(A_reg)
                    variance = np.dot(context, A_inv @ context)
                except np.linalg.LinAlgError:
                    variance = 1.0 # Fallback high variance
            
            bonus = self.c * np.sqrt(max(variance, 0.0))
            scores.append(np.dot(theta_hat, context) + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Apply forgetting factor and update with new observation
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
