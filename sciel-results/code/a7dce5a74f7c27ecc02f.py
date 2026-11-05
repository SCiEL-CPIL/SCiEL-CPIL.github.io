"""Bayesian Linear Bandit with Thompson Sampling and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Bayesian Linear Bandit with Thompson Sampling.
    
    Uses per-arm posterior distributions over weight vectors.
    Posterior covariance is decayed adaptively based on prediction residuals
    (Mahalanobis distance), allowing faster adaptation during regime shifts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Prior: N(0, alpha^2 * I)
        self.alpha_prior = 10.0
        self.A_inv = [self.alpha_prior ** 2 * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Track residuals for adaptive forgetting
        self.residuals = [[] for _ in range(n_arms)]
        self.max_residuals = 50
        
        # Base decay factor (for stable periods)
        self.base_decay = 0.99
        
        # Threshold for detecting regime shift
        self.shift_threshold = 3.0

    def _adaptive_decay(self, arm, context, reward):
        """
        Compute adaptive decay factor based on prediction residual.
        
        If the residual is large (indicating a regime shift), use a smaller
        decay factor (i.e., forget more quickly). If the residual is small,
        use a decay factor close to 1 (retain history).
        """
        if len(self.residuals[arm]) < 5:
            return self.base_decay
        
        # Compute prediction and residual
        # theta_hat = A_inv @ b
        theta_hat = self.A_inv[arm] @ self.b[arm]
        pred = theta_hat @ context
        residual = reward - pred
        
        # Compute prediction variance (diagonal of A_inv in context direction)
        pred_var = context @ self.A_inv[arm] @ context
        pred_std = np.sqrt(max(pred_var, 1e-10))
        
        # Mahalanobis-like normalized residual
        z = abs(residual) / pred_std
        
        # Store residual for tracking
        self.residuals[arm].append(z)
        if len(self.residuals[arm]) > self.max_residuals:
            self.residuals[arm].pop(0)
        
        # Recent average normalized residual
        recent_z = np.mean(self.residuals[arm][-10:]) if len(self.residuals[arm]) >= 10 else np.mean(self.residuals[arm])
        
        # If recent residuals are consistently large, accelerate forgetting
        if recent_z > self.shift_threshold:
            # Sharp increase in forgetting: decay = 0.90 to 0.95
            decay = max(0.85, 1.0 - 0.1 * min(recent_z / 10.0, 1.0))
            return decay
        else:
            # Normal forgetting
            return self.base_decay

    def select(self, context):
        """
        Thompson Sampling: sample from posterior for each arm, pick best.
        """
        context = np.asarray(context, dtype=np.float64)
        samples = []
        for a in range(self.n):
            # Sample theta ~ N(theta_hat, A_inv)
            theta_hat = self.A_inv[a] @ self.b[a]
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(self.A_inv[a] + 1e-10 * np.eye(self.d))
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
        
        # Compute adaptive decay factor
        decay = self._adaptive_decay(arm, context, reward)
        
        # Decay existing posterior (forgetting)
        self.A_inv[arm] = self.A_inv[arm] / decay  # Note: A_inv grows when we decay A
        # Actually, let's think about this more carefully.
        # A = sum(x_i x_i^T), A_inv = A^{-1}
        # If we decay A: A_new = decay * A_old + x x^T
        # Then A_inv_new = (decay * A_old + x x^T)^{-1}
        # Using matrix inversion lemma:
        # (decay * A_old + x x^T)^{-1} = (1/decay) * A_old^{-1} - (1/decay)^2 * A_old^{-1} x x^T A_old^{-1} / (1/decay + x^T A_old^{-1} x)
        
        # Simpler approach: maintain A_inv directly
        # A_old_inv = self.A_inv[arm]
        # A_new = decay * A_old + x x^T
        # A_new_inv = (decay * A_old + x x^T)^{-1}
        # Using Sherman-Morrison:
        # (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Let A = decay * A_old, u = x, v = x
        # (decay * A_old + x x^T)^{-1} = (decay * A_old)^{-1} - (decay * A_old)^{-1} x x^T (decay * A_old)^{-1} / (1 + x^T (decay * A_old)^{-1} x)
        # = (1/decay) * A_old^{-1} - (1/decay)^2 * A_old^{-1} x x^T A_old^{-1} / (1 + (1/decay) * x^T A_old^{-1} x)
        
        A_inv_old = self.A_inv[arm]
        b_old = self.b[arm]
        
        # Compute terms
        A_inv_x = A_inv_old @ context
        x_A_inv_x = context @ A_inv_x  # scalar
        
        # New A_inv
        factor = 1.0 / decay
        denom = 1.0 + factor * x_A_inv_x
        self.A_inv[arm] = factor * A_inv_old - (factor ** 2) * np.outer(A_inv_x, A_inv_x) / denom
        
        # New b
        # b_new = decay * b_old + reward * x
        self.b[arm] = decay * b_old + reward * context
        
        # Regularize to prevent numerical issues
        self.A_inv[arm] += 1e-10 * np.eye(self.d)
# EVOLVE-BLOCK-END
