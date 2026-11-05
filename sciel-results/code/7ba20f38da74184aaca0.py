"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Adaptive Thompson Sampling.
    
    Structural Change from Seed (LinUCB):
    1. Uses Thompson Sampling (Bayesian) instead of UCB (Deterministic).
    2. Adaptive Forgetting: Monitors prediction error to dynamically adjust the 
       forgetting factor or trigger resets, improving adaptation to non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0  # Regularization
        self.eye = np.eye(dim)
        
        # Initialize posterior parameters for each arm
        # A is precision matrix, b is precision vector
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor parameters
        # Wider range for adaptive forgetting: lower bound allows faster adaptation,
        # upper bound allows more stable learning in stationary periods
        self.beta_min = 0.85
        self.beta_max = 0.998
        self.beta = 0.99  # Current forgetting factor
        
        # Error tracking for adaptive beta
        self.error_history = []
        self.window_size = 15
        self.error_threshold = 1.2  # Lower threshold for more aggressive adaptation
        
        # Time management
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin
        
        # RNG for reproducibility in sampling if needed, but default is fine
        self.rng = np.random.default_rng()

    def _get_posterior_params(self, arm):
        """Compute mean and covariance for the arm's posterior."""
        A = self.A[arm]
        b = self.b[arm]
        
        # Add small jitter for numerical stability
        A_reg = A + 1e-6 * self.eye
        
        # Solve for mean: theta = A^{-1} b
        try:
            L = np.linalg.cholesky(A_reg)
            # theta = L^{-T} L^{-1} b
            # Solve L v = b
            v = np.linalg.solve(L, b)
            # Solve L^T theta = v
            theta = np.linalg.solve(L.T, v)
            
            # Return L for sampling
            return theta, L
        except np.linalg.LinAlgError:
            # Fallback to inverse if Cholesky fails
            inv_A = np.linalg.inv(A_reg)
            theta = inv_A @ b
            # For sampling, we need a matrix M such that M M^T = inv_A
            # Use Cholesky of inv_A if possible
            try:
                M = np.linalg.cholesky(inv_A)
                return theta, M.T  # Return M.T so that M.T @ z gives correct sample
            except np.linalg.LinAlgError:
                # Use eigendecomposition as last resort
                eigvals, eigvecs = np.linalg.eigh(inv_A)
                eigvals = np.maximum(eigvals, 1e-8)
                M = eigvecs @ np.diag(np.sqrt(eigvals))
                return theta, M

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a safe default
            # Use the mean of the arm with highest estimated reward
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta, _ = self._get_posterior_params(a)
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    pass
            return best_arm

        scores = []
        for a in range(self.n):
            theta, M = self._get_posterior_params(a)
            
            # Sample theta_a from posterior
            # M is such that M M^T = A^{-1} (or M^T M = A^{-1} depending on fallback)
            # We want y ~ N(0, A^{-1}), so y = M @ z where z ~ N(0, I)
            z = self.rng.normal(0, 1, self.d)
            try:
                theta_sample = theta + M @ z
            except:
                theta_sample = theta

            score = theta_sample @ context
            scores.append(score)

        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update with forgetting factor
        beta = self.beta
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context
        
        # Adaptive Forgetting Logic
        # Compute prediction error
        try:
            theta, _ = self._get_posterior_params(arm)
            pred = theta @ context
            error = reward - pred
            self.error_history.append(error**2)
            
            # Keep history bounded
            if len(self.error_history) > self.window_size:
                self.error_history.pop(0)
            
            # Calculate average error
            avg_error = np.mean(self.error_history)
            
            # If error is high, we are in a non-stationary regime or have bad model.
            # We should forget faster (lower beta) to adapt.
            # If error is low, we can trust our model more (higher beta).
            
            # Map avg_error to beta with smooth interpolation
            # Use a smooth mapping: beta = beta_max - (beta_max - beta_min) * min(1, avg_error / (2 * threshold))
            norm_error = min(1.0, avg_error / (2.0 * self.error_threshold))
            self.beta = self.beta_max - (self.beta_max - self.beta_min) * norm_error
            
            # Hard reset if error is extremely high
            if abs(error) > 4.0:
                # Reset to prior
                self.A[arm] = np.eye(self.d) * self.lam
                self.b[arm] = np.zeros(self.d)
                self.error_history = []
                
        except Exception:
            pass
# EVOLVE-BLOCK-END
