"""Bayesian Linear Bandit with Thompson Sampling and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Bayesian Linear Bandit with Thompson Sampling and adaptive forgetting.
    
    Uses per-arm posterior distributions over weight vectors.
    Posterior covariance is decayed adaptively based on the Mahalanobis distance
    of the latest prediction residual. This allows the policy to sharply reduce
    uncertainty and accelerate weight adaptation specifically when prediction 
    errors indicate a regime shift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Prior: N(0, alpha^2 * I)
        self.alpha_prior = 10.0
        self.A_inv = [self.alpha_prior ** 2 * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Base decay factor (for stable periods)
        self.base_decay = 0.99
        
        # Minimum decay factor (maximum forgetting rate)
        self.min_decay = 0.70
        
        # Regularization for numerical stability
        self.reg = 1e-6

    def select(self, context):
        """
        Thompson Sampling: sample from posterior for each arm, pick best.
        """
        context = np.asarray(context, dtype=np.float64)
        samples = []
        for a in range(self.n):
            # Sample theta ~ N(theta_hat, A_inv)
            theta_hat = self.A_inv[a] @ self.b[a]
            
            # Ensure A_inv is symmetric
            A_inv_sym = (self.A_inv[a] + self.A_inv[a].T) / 2.0
            
            # Use Cholesky decomposition for sampling
            # Add small regularization to ensure PD
            try:
                L = np.linalg.cholesky(A_inv_sym + self.reg * np.eye(self.d))
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition to fix PD issues
                try:
                    eigvals, eigvecs = np.linalg.eigh(A_inv_sym)
                    eigvals = np.maximum(eigvals, self.reg)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                except np.linalg.LinAlgError:
                    L = np.eye(self.d) * np.sqrt(self.alpha_prior ** 2)
            
            # Sample
            z = np.random.standard_normal(self.d)
            theta_sample = theta_hat + L @ z
            
            # Predicted reward
            reward_sample = theta_sample @ context
            samples.append(reward_sample)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """
        Update posterior using Bayesian linear regression update
        with adaptive forgetting.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute adaptive decay factor based on prediction residual
        # Calculate current mean and variance
        theta_hat = self.A_inv[arm] @ self.b[arm]
        pred = theta_hat @ context
        
        # Prediction variance (diagonal of A_inv in context direction)
        # A_inv is symmetric, so context @ A_inv @ context
        pred_var = context @ (self.A_inv[arm] @ context)
        pred_std = np.sqrt(max(pred_var, 1e-10))
        
        # Residual
        residual = reward - pred
        
        # Mahalanobis-like normalized residual
        z = abs(residual) / pred_std
        
        # Adaptive decay:
        # If z is small, decay is close to base_decay (retain history)
        # If z is large, decay decreases (forget more)
        # Mapping: z=0 -> base_decay, z=10 -> min_decay
        # Linear interpolation in the "forgetting rate"
        # Forgetting rate = 1 - decay
        # base_forget = 1 - base_decay
        # max_forget = 1 - min_decay
        # forget_rate = min(max_forget, base_forget + (max_forget - base_forget) * (z / 10.0))
        # decay = 1 - forget_rate
        
        base_forget = 1.0 - self.base_decay
        max_forget = 1.0 - self.min_decay
        
        # Scale z to [0, 1] range for interpolation, saturating at 10
        z_scaled = min(z / 10.0, 1.0)
        
        # Interpolate forgetting rate
        forget_rate = base_forget + (max_forget - base_forget) * z_scaled
        decay = 1.0 - forget_rate
        
        # Ensure decay is within [min_decay, 1.0]
        decay = max(self.min_decay, min(decay, 1.0))
        
        # 2. Update A_inv and b using Sherman-Morrison with decay
        A_inv_old = self.A_inv[arm]
        b_old = self.b[arm]
        
        # Compute terms
        A_inv_x = A_inv_old @ context
        x_A_inv_x = context @ A_inv_x  # scalar
        
        # New A_inv
        # A_new = decay * A_old + x x^T
        # A_new_inv = (decay * A_old)^{-1} - (decay * A_old)^{-1} x x^T (decay * A_old)^{-1} / (1 + x^T (decay * A_old)^{-1} x)
        # (decay * A_old)^{-1} = (1/decay) * A_old^{-1}
        factor = 1.0 / decay
        denom = 1.0 + factor * x_A_inv_x
        
        # Sherman-Morrison update
        self.A_inv[arm] = factor * A_inv_old - (factor ** 2) * np.outer(A_inv_x, A_inv_x) / denom
        
        # New b
        # b_new = decay * b_old + reward * x
        self.b[arm] = decay * b_old + reward * context
        
        # 3. Numerical Regularization
        # Symmetrize to prevent drift. 
        # Note: We do NOT add regularization here as it would increase posterior variance 
        # indefinitely, counteracting the decay mechanism. The decay itself provides 
        # a natural bound on the effective sample size.
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
# EVOLVE-BLOCK-END
