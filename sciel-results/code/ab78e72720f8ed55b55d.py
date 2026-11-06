"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear bandit using Discounted Thompson Sampling (DTS).
    
    This approach maintains a posterior distribution over the weight vector for each arm.
    The posterior is updated with a decay factor to handle non-stationarity.
    Selection is done by sampling from the posterior and picking the arm with the highest predicted reward.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Decay factor for non-stationarity
        
        # Per-arm state: A (precision matrix) and b (mean vector scaled by precision)
        # Prior: N(0, I) => A = I, b = 0
        self.A = [np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        
        # Random generator
        self.rng = np.random.default_rng(42)

    def select(self, context):
        """Select arm by sampling from the posterior distribution."""
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n):
            # Compute mean
            # mu = A^{-1} b
            try:
                mu = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                mu = np.zeros(self.d)
                
            # Get Cholesky decomposition of A to sample from N(mu, A^{-1})
            # If A is not positive definite due to numerical issues, add jitter
            try:
                L = np.linalg.cholesky(self.A[a])
            except np.linalg.LinAlgError:
                # Add small jitter to diagonal
                self.A[a] = self.A[a].copy()
                self.A[a] += 1e-6 * np.eye(self.d)
                try:
                    L = np.linalg.cholesky(self.A[a])
                except np.linalg.LinAlgError:
                    # Fallback to identity if still failing
                    L = np.eye(self.d)
            
            # Sample from N(0, I)
            z = self.rng.standard_normal(self.d)
            
            # Solve L^T v = z  => v = (L^T)^{-1} z
            # This ensures Cov(v) = A^{-1}
            try:
                v = np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                v = np.zeros(self.d)
                
            theta_sample = mu + v
            
            score = np.dot(context, theta_sample)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        """Update the posterior for the selected arm with decay."""
        # Decay the current state
        self.A[arm] = self.gamma * self.A[arm]
        self.b[arm] = self.gamma * self.b[arm]
        
        # Update with new observation
        # A_new = A_old + x x^T
        # b_new = b_old + r x
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

# EVOLVE-BLOCK-END
