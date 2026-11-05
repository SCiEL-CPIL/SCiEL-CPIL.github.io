"""Bayesian Linear Bandit with Adaptive Forgetting via Residual Detection."""
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
        # Prior: N(0, I)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.V_inv = [np.eye(dim) for _ in range(n_arms)]
        self.V = [np.eye(dim) for _ in range(n_arms)]  # For efficient inverse updates
        
        # Adaptive forgetting parameters
        self.base_decay = 0.99
        self.shift_decay = 0.5
        self.residual_threshold = 2.0  # z-score threshold for regime shift detection
        
        # Track prediction errors for adaptive forgetting
        self.last_prediction = [None] * n_arms
        self.last_context = [None] * n_arms
        
        # Time tracking for safety
        self.start_time = time.time()
        self.deadline = 140.0  # 10s margin before 150s kill

    def _mahalanobis_distance(self, x, V):
        """Compute Mahalanobis distance squared: x^T V x"""
        return x @ (V @ x)

    def select(self, context):
        # Safety check
        if time.time() - self.start_time > self.deadline:
            # Fallback to best known arm (simple heuristic: pick arm 0 or random)
            # In practice, we should return a valid arm
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            # Thompson Sampling: sample theta from posterior N(theta, V)
            # V = (V_inv)^(-1)
            # We maintain V_inv, so V = inv(V_inv)
            # To avoid explicit inversion, we can use Cholesky decomposition
            # theta_sample = theta + L @ z, where V = L L^T
            try:
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d)
                theta_sample = self.theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (should not happen with PD matrices)
                theta_sample = self.theta[a]
            
            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Safety check
        if time.time() - self.start_time > self.deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Adaptive Forgetting Detection
        # Compute prediction error
        if self.last_context[arm] is not None:
            # This is a new context, so we can't directly compare residuals in the same context
            # Instead, we check if the residual from the last update was large
            # Actually, standard approach: check residual at the time of the last prediction
            # But we didn't store the reward then. 
            # Alternative: check if the current residual (reward - theta @ context) is large
            # relative to the uncertainty.
            
            # Compute current residual
            pred = self.theta[arm] @ context
            residual = reward - pred
            
            # Compute uncertainty (std) of the prediction
            # var = context^T V context
            var = self._mahalanobis_distance(context, self.V[arm])
            std = np.sqrt(max(var, 1e-8))
            
            # z-score
            z_score = abs(residual) / std
            
            # Determine decay factor
            if z_score > self.residual_threshold:
                decay = self.shift_decay
            else:
                decay = self.base_decay
        else:
            decay = 1.0  # No forgetting on first observation for this arm
        
        # 2. Apply Forgetting
        # Decay the information: V_inv = decay * V_inv
        # This effectively shrinks the confidence, allowing new data to dominate
        self.V_inv[arm] *= decay
        # Update V = inv(V_inv)
        # Instead of inverting every time, we can use the update rule with decay
        # But since we decay V_inv, we need to update V accordingly.
        # V_new = inv(decay * V_old_inv) = (1/decay) * V_old
        # So V scales by 1/decay.
        self.V[arm] /= decay
        
        # 3. Update Posterior (Bayesian Linear Regression Update)
        # New V_inv = V_inv + x x^T
        # New V = (V_inv)^(-1)
        # Using matrix inversion lemma:
        # V_new = V - V x (x^T V x + 1)^(-1) x^T V
        # theta_new = theta + V_new (x (r - theta^T x))
        
        x = context
        V = self.V[arm]
        
        # Compute terms
        Vx = V @ x
        denom = x @ Vx + 1.0
        V_new = V - np.outer(Vx, Vx) / denom
        
        # Update theta
        # theta_new = theta + V_new @ x * (r - theta^T x)
        # Note: (r - theta^T x) is the residual
        residual = reward - self.theta[arm] @ x
        self.theta[arm] = self.theta[arm] + (V_new @ x) * residual
        
        # Update stored matrices
        self.V[arm] = V_new
        self.V_inv[arm] = np.linalg.inv(V_new)
        
        # Update tracking
        self.last_context[arm] = context
        self.last_prediction[arm] = self.theta[arm] @ context
# EVOLVE-BLOCK-END
