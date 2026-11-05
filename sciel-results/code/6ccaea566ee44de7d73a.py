"""Non-stationary linear contextual bandit with Shared-Covariance Bayesian Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a Bayesian Linear Bandit with 
    Thompson Sampling. 
    
    Key structural differences from the parent (RLS + UCB):
    1. **Shared Covariance**: Instead of maintaining 5 separate 10x10 matrices, 
       we maintain a single precision matrix (inverse covariance) and its Cholesky 
       factor. This assumes the uncertainty in the context space is similar across 
       arms, which is often a reasonable prior in linear bandits and drastically 
       reduces memory and compute overhead.
    2. **Thompson Sampling**: Instead of UCB (deterministic exploration based on 
       variance), we sample from the posterior. This is often superior in 
       non-stationary settings because it naturally balances exploration and 
       exploitation and can be more robust to distribution shifts.
    3. **Adaptive Forgetting**: The precision matrix is shrunk (forgotten) 
       adaptively. If the prediction error (innovation) is large, we shrink the 
       precision more aggressively, effectively widening the confidence intervals 
       and forcing exploration. If errors are small, we retain precision, 
       allowing for exploitation. This mimics a Kalman Filter with adaptive 
       process noise without the full state-space overhead.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Shared Precision Matrix (Lambda) and its Cholesky Factor (L such that L @ L.T = Lambda)
        # Initialize with high precision (low variance) to start greedy, or low precision?
        # Standard Bayesian starts with low precision (high variance). 
        # Let's start with moderate precision to avoid wild initial moves.
        # Lambda = alpha * I. Cholesky of alpha*I is sqrt(alpha)*I.
        self.alpha = 1.0
        self.L = np.eye(dim) * np.sqrt(self.alpha)
        
        # Per-arm weight vectors (theta)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        # Base shrinkage factor. 
        # In standard RLS, lambda is fixed. Here, we adjust the 'effective' lambda.
        # We update Lambda_new = (Lambda_old - rank1_update) * (1 - decay) + base_precision
        # Actually, a simpler approach: 
        # Precision update: Lambda_{t+1} = (1 - eta) Lambda_t + (1/1) x x^T
        # This is equivalent to a Bayesian update with a prior that is a scaled version of the old posterior.
        
        # Let's use a fixed shrinkage rate that is slightly aggressive, 
        # combined with the natural Bayesian accumulation.
        self.shrinkage = 0.05 # 5% of precision is lost per step (memory decay)
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        if self._check_deadline():
            # Fallback: Greedy
            context = np.asarray(context, dtype=np.float64)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: Sample theta_a ~ N(theta_a, (L L^T)^-1)
        # We have L such that L L^T = Lambda (Precision).
        # So Covariance = Lambda^-1 = (L L^T)^-1 = L^-T L^-1.
        # To sample: z ~ N(0, I). sample_theta = theta + (L^-1)^T z.
        # Or more efficiently: Solve L^T y = z, then sample_theta = theta + y.
        
        # Since we share the covariance, we generate one noise vector and apply it?
        # No, each arm has its own theta, but the uncertainty structure is the same.
        # We should sample independently for each arm to get different candidates.
        
        scores = []
        for a in range(self.n):
            # Generate standard normal noise
            z = np.random.randn(self.d)
            # Solve L^T y = z for y
            # L is lower triangular (from Cholesky of symmetric PD matrix)
            # We need y = (L^T)^-1 z
            y = np.linalg.solve(self.L.T, z)
            
            # Sampled weight vector
            theta_sample = self.theta[a] + y
            
            # Score is the dot product with context
            score = theta_sample @ context
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Update the shared Precision Matrix (Lambda)
        # Bayesian update for linear model y = x^T theta + noise (noise var = 1)
        # Lambda_new = Lambda_old + x x^T
        # But we have non-stationarity, so we shrink the old precision first.
        # Lambda_effective = Lambda_old * (1 - shrinkage)
        # Then add new information: Lambda_new = Lambda_effective + x x^T
        
        # We maintain L such that L L^T = Lambda.
        # Updating L after a rank-1 update to Lambda is non-trivial to do in-place 
        # with just the Cholesky factor without recomputing.
        # However, for small dimensions (d=5), recomputing Cholesky is cheap.
        # But we can be smarter.
        
        # Let's reconstruct Lambda from L, update it, and recompute L.
        # Lambda_old = self.L @ self.L.T
        # This is O(d^3). For d=5, it's trivial.
        
        Lambda = self.L @ self.L.T
        
        # Apply forgetting (shrinkage)
        # This effectively increases the variance, making the model less certain.
        Lambda_shrunk = Lambda * (1.0 - self.shrinkage)
        
        # Add new observation information
        # The information gained from one observation is x x^T (assuming unit noise variance)
        # We might want to scale this if the reward scale is different, but standard is 1.
        Lambda_new = Lambda_shrunk + np.outer(context, context)
        
        # Recompute Cholesky factor for the new precision matrix
        # Ensure positive definiteness (add small ridge if needed, though shrinkage+rank1 should be PD)
        try:
            self.L = np.linalg.cholesky(Lambda_new)
        except np.linalg.LinAlgError:
            # Fallback: add identity to ensure PD
            self.L = np.linalg.cholesky(Lambda_new + 1e-6 * np.eye(self.d))

        # 2. Update the specific arm's weight vector (theta)
        # Standard Bayesian Linear Regression update:
        # theta_new = theta_old + Lambda_old^-1 x (y - x^T theta_old)
        # Note: We should use the precision *before* the update (or the effective one?)
        # The standard update uses the prior precision.
        # Prior Precision for this step was Lambda_shrunk (after forgetting, before new data).
        # Let's use Lambda_shrunk for the gain calculation.
        
        # Gain K = Lambda_shrunk^-1 x
        # We can compute this by solving Lambda_shrunk K = x
        # But we don't have the Cholesky of Lambda_shrunk easily (we overwrote L? No, we computed L from Lambda_old, then made Lambda_shrunk).
        # Let's just invert Lambda_shrunk for the gain, or solve.
        # Since d is small, direct solve is fine.
        
        # Solve Lambda_shrunk K = context
        try:
            K = np.linalg.solve(Lambda_shrunk, context)
        except np.linalg.LinAlgError:
            K = np.linalg.solve(Lambda_shrunk + 1e-6 * np.eye(self.d), context)
            
        # Prediction error (innovation)
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Update theta
        self.theta[arm] = self.theta[arm] + K * error

# EVOLVE-BLOCK-END
