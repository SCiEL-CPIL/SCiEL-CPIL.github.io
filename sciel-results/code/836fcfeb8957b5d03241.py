"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Thompson Sampling with adaptive covariance inflation for non-stationarity."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Posterior mean and covariance per arm
        self.theta_mean = [np.zeros(dim) for _ in range(n_arms)]
        self.V = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # covariance
        self.V_inv = [np.eye(dim) for _ in range(n_arms)]     # inverse covariance
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A = [np.eye(dim) for _ in range(n_arms)]
        # Adaptive inflation
        self.alpha = 0.995
        self.base_inflation = 1.005
        self.max_inflation = 3.0
        self.min_inflation = 1.0
        self.err_threshold = 2.0
        self.err_decay = 0.9
        self.recent_errors = [0.0 for _ in range(n_arms)]
        self._rng = np.random.default_rng(42)

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Sample theta from N(theta_mean, V)
            # Use Cholesky for efficiency
            try:
                L = np.linalg.cholesky(self.V[a])
                z = self._rng.standard_normal(self.d)
                theta_sample = self.theta_mean[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                theta_sample = self.theta_mean[a]
            scores.append(float(theta_sample @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update posterior mean and covariance
        # Standard Bayesian linear regression update
        A = self.A[arm]
        b = self.b[arm]
        
        # Compute prediction and innovation
        pred = float(self.theta_mean[arm] @ context)
        innovation = reward - pred
        
        # Compute prediction variance for this context
        try:
            pred_var = float(context @ self.V[arm] @ context)
        except:
            pred_var = 1.0
        
        # Adaptive inflation based on standardized innovation
        std_innov = abs(innovation) / (np.sqrt(pred_var) + 1e-8)
        self.recent_errors[arm] = max(self.recent_errors[arm] * self.err_decay, std_innov)
        
        # Inflation factor: larger when errors are large
        if self.recent_errors[arm] > self.err_threshold:
            inflation = min(self.max_inflation, self.base_inflation + 
                           (self.recent_errors[arm] - self.err_threshold) * 0.3)
        else:
            inflation = self.base_inflation
        
        # Update A and b with forgetting
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
        # Update V and V_inv using matrix inversion lemma with inflation
        # New V = inflation * (V - V @ x @ x^T @ V / (1 + x^T @ V @ x))
        # But with forgetting, we need to be careful. Let's use the A, b formulation.
        # Actually, let's maintain V directly with the inflation.
        
        # Compute V update
        # V_new = inflation * (V - V @ x @ x^T @ V / (1 + x^T @ V @ x))
        x = context
        V = self.V[arm]
        xVx = float(x @ V @ x)
        denom = 1.0 + xVx
        # V @ x
        Vx = V @ x
        # V - Vx @ x^T / denom
        V_new = V - np.outer(Vx, x) / denom
        V_new = inflation * V_new
        
        # Ensure symmetry
        V_new = (V_new + V_new.T) / 2.0
        self.V[arm] = V_new
        
        # Update V_inv
        # V_inv_new = (1/inflation) * (V_inv + V_inv @ x @ x^T @ V_inv / (1 + x^T @ V @ x))
        V_inv = self.V_inv[arm]
        V_inv_x = V_inv @ x
        V_inv_new = V_inv + np.outer(V_inv_x, x) / denom
        V_inv_new = V_inv_new / inflation
        V_inv_new = (V_inv_new + V_inv_new.T) / 2.0
        self.V_inv[arm] = V_inv_new
        
        # Update theta_mean
        # theta_mean_new = theta_mean + V @ x @ (reward - x^T @ theta_mean) / (1 + x^T @ V @ x)
        # But with inflation, we need to adjust
        # Actually, with the inflated V, the update should use the pre-inflation V for the gain
        # Let's recompute: the standard update uses the prior V
        gain = V @ x / denom
        self.theta_mean[arm] = self.theta_mean[arm] + gain * innovation
# EVOLVE-BLOCK-END
