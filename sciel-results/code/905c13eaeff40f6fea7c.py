"""Differential Linear Bandit for non-stationary contexts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Differential Linear Bandit.
    
    Models reward as r = x^T w_global + x^T theta_arm + noise.
    - w_global: common drift/trend, updated with all observations.
    - theta_arm: arm-specific preference, updated with de-trended residuals.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        
        # Global trend model (shared across arms)
        # We use a ridge regression style update for stability
        # A_g and b_g for global model
        self.A_g = np.eye(dim) * 10.0  # Regularization to keep it stable
        self.b_g = np.zeros(dim)
        
        # Per-arm models for the residual (de-trended) component
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Hyperparameters
        self.alpha = 1.0  # Exploration coefficient
        self.lambda_g = 0.99 # Forgetting factor for global trend (slower adaptation to be stable)
        self.lambda_arm = 0.995 # Forgetting factor for arm models (faster adaptation to arm-specific changes)
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0 # 5 seconds safety margin before 150s kill

    def _compute_global_theta(self):
        """Compute the current global trend vector."""
        try:
            return np.linalg.solve(self.A_g, self.b_g)
        except np.linalg.LinAlgError:
            return np.zeros(self.dim)

    def select(self, context):
        """
        Select an arm based on UCB of the total estimated reward.
        Estimated reward = x^T w_global + x^T theta_arm
        """
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback to a simple heuristic if we are running out of time
            # Just pick a random arm or the one with highest past average
            return int(np.random.randint(self.n_arms))

        context = np.asarray(context)
        w_global = self._compute_global_theta()
        
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            # Mean estimate: x^T w_global + x^T theta_a
            inv_A = np.linalg.inv(self.A[a])
            theta_a = inv_A @ self.b[a]
            
            mu_a = context @ w_global + context @ theta_a
            
            # Uncertainty: based on the arm-specific model
            # We assume the global trend is known (or its uncertainty is small relative to arm-specific uncertainty)
            # A more rigorous approach would add global uncertainty, but for now we focus on arm-specific
            variance = context @ inv_A @ context
            ucb = mu_a + self.alpha * np.sqrt(variance)
            
            scores[a] = ucb
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update global trend and arm-specific model.
        """
        context = np.asarray(context)
        
        # 1. Update Global Trend Model
        # The global model tries to predict the "average" reward.
        # We update it with the current observation.
        # Using a forgetting factor to handle non-stationarity in the trend.
        self.A_g = self.lambda_g * self.A_g + np.outer(context, context)
        self.b_g = self.lambda_g * self.b_g + reward * context
        
        # 2. Compute De-trended Residual
        # Use the *previous* global estimate to avoid using current reward in its own residual
        # But for simplicity and stability, using the updated one is often done in online settings.
        # Let's use the estimate *before* this update was applied to the global model?
        # Actually, standard practice in differential bandits is to use the current global estimate.
        # Let's compute w_global based on the state *before* this update to be strictly causal?
        # Or just use the updated one. Given the small step size, the difference is minor.
        # Let's use the updated global theta for the residual calculation.
        w_global = self._compute_global_theta()
        
        residual = reward - context @ w_global
        
        # 3. Update Arm-Specific Model
        # Update with the residual reward
        self.A[arm] = self.lambda_arm * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.lambda_arm * self.b[arm] + residual * context
        
        self.t += 1
# EVOLVE-BLOCK-END
