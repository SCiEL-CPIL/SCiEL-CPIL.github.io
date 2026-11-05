"""Rewritten program using Discounted Thompson Sampling for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg

class Policy:
    """
    Discounted Thompson Sampling for non-stationary linear contextual bandits.
    
    Instead of a Kalman filter with adaptive process noise, this policy uses
    Discounted Least Squares (DLS) to maintain a Bayesian posterior over the
    weight vectors. Older observations are exponentially discounted, which
    allows the policy to adapt to changing arm weights.
    
    Exploration is performed via Thompson Sampling: at each step, a sample
    is drawn from the posterior distribution of each arm's weights, and the
    arm with the highest predicted reward for the sample is selected.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # seconds, safety margin before 120 s kill

        # Discount factor for non-stationarity. 
        # Lower values (e.g., 0.9) react faster to changes but forget faster.
        # Higher values (e.g., 0.99) are more stable but slower to adapt.
        self.gamma = 0.95
        
        # Regularization parameter for numerical stability (prior precision)
        self.lambda_reg = 1.0

        # Per-arm precision matrices (V) and sum vectors (b)
        # V is the precision matrix (inverse covariance) of the posterior.
        # b is the numerator for the mean calculation: mu = V^-1 b.
        self.V = [np.eye(dim) * self.lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Counters for potential adaptive logic (not used in basic DTS)
        self.count = [0 for _ in range(n_arms)]

        # Random state for reproducibility if needed, though standard np.random is fine
        self.rng = np.random.default_rng(42)

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest mean theta @ context
            # Compute means quickly
            scores = []
            for a in range(self.n):
                try:
                    mu = np.linalg.solve(self.V[a], self.b[a])
                except np.linalg.LinAlgError:
                    mu = np.zeros(self.d)
                scores.append(mu @ context)
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            # Compute posterior mean: mu = V^-1 b
            try:
                mu = np.linalg.solve(self.V[a], self.b[a])
            except np.linalg.LinAlgError:
                mu = np.zeros(self.d)
            
            # Compute posterior covariance: Sigma = V^-1
            # We sample theta ~ N(mu, V^-1)
            # Using Cholesky decomposition of V: V = L L^T
            # Then V^-1 = L^-T L^-1
            # We want to sample z ~ N(0, I) and set theta = mu + L^-T z
            # L^-T z is the solution w to L^T w = z
            
            try:
                L = np.linalg.cholesky(self.V[a])
                z = self.rng.standard_normal(self.d)
                # Solve L^T w = z for w
                w = scipy.linalg.solve_triangular(L.T, z, lower=False)
                theta_sample = mu + w
            except np.linalg.LinAlgError:
                # Fallback to mean if Cholesky fails
                theta_sample = mu

            score = theta_sample @ ctx
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Update precision and sum with discounting
        # V_new = gamma * V_old + x x^T
        # b_new = gamma * b_old + r x
        
        self.V[arm] = self.gamma * self.V[arm] + np.outer(ctx, ctx)
        self.b[arm] = self.gamma * self.b[arm] + reward * ctx

        # Symmetrize V to prevent drift
        self.V[arm] = 0.5 * (self.V[arm] + self.V[arm].T)

        self.count[arm] += 1
# EVOLVE-BLOCK-END
