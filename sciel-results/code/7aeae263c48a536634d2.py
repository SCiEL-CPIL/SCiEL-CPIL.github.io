"""Bayesian Linear Bandit with Adaptive Mahalanobis Forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Thompson Sampling for non-stationary linear bandits.
    
    Uses a Bayesian framework where the posterior over weights is Gaussian.
    The covariance matrix is updated using the matrix inversion lemma for efficiency.
    An adaptive forgetting mechanism is triggered by large Mahalanobis residuals,
    which indicates a potential regime shift in the underlying weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize posterior parameters for each arm
        # P: Posterior covariance (d x d)
        # theta: Posterior mean (d,)
        # We start with a uniform prior: N(0, I)
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Noise variance assumption (standard for Gaussian rewards)
        self.sigma2 = 1.0
        
        # Adaptive forgetting parameters
        # Base decay rate (close to 1 means little forgetting)
        self.base_decay = 0.99
        # Threshold for Mahalanobis distance to trigger stronger forgetting
        self.mahalanobis_threshold = 3.0
        # Maximum decay reduction (how much we can shrink the window effectively)
        # If residual is huge, we want to forget more aggressively.
        # decay_factor will be in [min_decay, 1.0]
        self.min_decay = 0.8
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0 # Safety margin

    def _mahalanobis_distance(self, context, arm):
        """Calculate Mahalanobis distance of the context w.r.t arm's posterior."""
        # d = ||x||_P = sqrt(x^T P x)
        Px = self.P[arm] @ context
        return np.sqrt(np.dot(context, Px))

    def select(self, context):
        # Check time limit
        if time.time() > self.deadline:
             # Fallback to best arm based on current means if we are running out of time
             # This should rarely happen if the algorithm is fast enough
             scores = [np.dot(self.theta[a], context) for a in range(self.n)]
             return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Sample from posterior: theta_a ~ N(theta_mean, P)
            # We can sample directly: theta_sample = theta_mean + P @ z / ||z|| * sqrt(diag(P))?
            # More efficient: Cholesky decomposition or SVD.
            # Since d is likely small (5 arms, dim usually 5-10), Cholesky is fine.
            try:
                L = np.linalg.cholesky(self.P[a])
                z = np.random.randn(self.d)
                theta_sample = self.theta[a] + L @ z
                scores.append(np.dot(theta_sample, context))
            except np.linalg.LinAlgError:
                # Fallback if covariance becomes singular or ill-conditioned
                scores.append(np.dot(self.theta[a], context))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check time limit
        if time.time() > self.deadline:
            return

        # 1. Compute prediction and residual
        predicted_reward = np.dot(self.theta[arm], context)
        residual = reward - predicted_reward
        
        # 2. Compute Mahalanobis distance of the residual
        # The variance of the prediction is context^T P context + sigma^2
        # The "standardized" residual (z-score) is residual / sqrt(context^T P context + sigma^2)
        pred_var = np.dot(context, self.P[arm] @ context) + self.sigma2
        std_residual = residual / np.sqrt(pred_var)
        
        # 3. Determine adaptive decay factor
        # If |std_residual| is large, it suggests the model is wrong (regime shift).
        # We want to increase the effective forgetting (lower the decay factor) to adapt faster.
        # Normal decay: 0.99 (slow forgetting)
        # If |std_residual| > threshold, we drop decay to min_decay (e.g., 0.8)
        
        abs_std_res = abs(std_residual)
        if abs_std_res > self.mahalanobis_threshold:
            # Interpolate between base_decay and min_decay based on severity
            # Severity: how much above threshold
            # Let's use a simple linear mapping or exponential decay of the "confidence"
            # Confidence in current model drops as residual grows.
            # decay = base_decay - (base_decay - min_decay) * min(1, (abs_std_res - threshold) / threshold)
            excess = abs_std_res - self.mahalanobis_threshold
            # Normalize excess, cap at 1.0 for the interpolation factor
            interp_factor = min(1.0, excess / self.mahalanobis_threshold)
            decay = self.base_decay - (self.base_decay - self.min_decay) * interp_factor
        else:
            decay = self.base_decay

        # 4. Update Posterior using Matrix Inversion Lemma
        # Standard Bayesian update for linear regression:
        # P_new = (P_old^-1 + x x^T / sigma^2)^-1
        # theta_new = P_new (P_old^-1 theta_old + x * r / sigma^2)
        
        # To incorporate forgetting, we first decay the existing information.
        # Decaying P (covariance) increases it (more uncertainty).
        # Decaying theta (mean) moves it towards 0 (prior mean).
        # A common way to implement "forgetting" in Bayesian linear models is to
        # shrink the precision (inverse covariance) and the sufficient statistics.
        
        # Let's work with precision matrices V = P^-1 and natural parameters.
        # However, maintaining P and theta directly is easier for sampling.
        # Let's use the update rule with decay applied to the *information* accumulated so far.
        
        # Alternative approach: Update P and theta as if we are adding a new data point,
        # but first scale the existing P and theta to reflect forgetting.
        
        # If we multiply P by 1/decay, we increase uncertainty.
        # If we multiply theta by decay, we shrink the estimate towards 0.
        # This is equivalent to having a prior that decays.
        
        # Let's try:
        # P_old_effective = P / decay
        # theta_old_effective = theta * decay
        
        P_old = self.P[arm]
        theta_old = self.theta[arm]
        
        P_eff = P_old / decay
        theta_eff = theta_old * decay
        
        # Now perform standard Bayesian update with x, r
        # P_new = (P_eff^-1 + x x^T / sigma^2)^-1
        # Using Sherman-Morrison:
        # (A + u v^T)^-1 = A^-1 - A^-1 u v^T A^-1 / (1 + v^T A^-1 u)
        # Here A = P_eff^-1, u = x/sqrt(sigma2), v = x/sqrt(sigma2)
        # Actually, it's easier to use the formula:
        # P_new = P_eff - P_eff x (sigma2 + x^T P_eff x)^-1 x^T P_eff
        
        x = context
        Px = P_eff @ x
        denom = self.sigma2 + np.dot(x, Px)
        
        # Update P
        self.P[arm] = P_eff - np.outer(Px, Px) / denom
        
        # Update theta
        # theta_new = theta_eff + P_new x (r - x^T theta_eff) / denom
        # Or simpler: theta_new = theta_eff + P_eff x (r - x^T theta_eff) / denom
        # Note: P_new x = P_eff x - P_eff x x^T P_eff x / denom
        # theta_new = theta_eff + P_new x * (r - x^T theta_eff) / denom
        # Let's use the direct formula:
        theta_new = theta_eff + (Px / denom) * (reward - np.dot(x, theta_eff))
        self.theta[arm] = theta_new
        
        # Symmetrize P to prevent numerical drift
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

# EVOLVE-BLOCK-END
