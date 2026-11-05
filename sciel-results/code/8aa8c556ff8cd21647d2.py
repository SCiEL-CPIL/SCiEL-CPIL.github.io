import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit with Adaptive Kalman Filter.
    Uses per-arm linear models with rank-1 covariance updates.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm parameters
        # Sigma is covariance (d x d)
        # theta is mean (d,)
        self.Sigma = [np.eye(self.d) * 1.0 for _ in range(self.n)]
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        
        # Adaptive Kalman parameters
        self.Q_min = 0.01
        self.Q_max = 1.0
        self.R = 1.0 # Observation noise
        self.error_scale = 1.0
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0 # Safety margin before 120s/150s kill

    def select(self, context):
        if time.time() > self.deadline:
            return 0
            
        context = np.array(context, dtype=float)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Prediction
            mu = self.theta[a]
            pred = mu @ context
            # Variance
            var = context @ self.Sigma[a] @ context
            # UCB
            scores[a] = pred + 1.0 * np.sqrt(var)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.array(context, dtype=float)
        
        # 1. Calculate Prediction Error
        mu = self.theta[arm]
        pred = mu @ context
        error = reward - pred
        
        # 2. Adaptive Process Noise (Q)
        # If error is large, assume drift is happening, increase Q (forget more)
        # If error is small, assume stable, decrease Q
        abs_err = abs(error)
        # Map error to Q range
        # Use a smooth function or simple clip
        q_val = self.Q_min + (self.Q_max - self.Q_min) * np.tanh(abs_err / self.error_scale)
        
        # 3. Kalman Filter Update
        Sigma_prior = self.Sigma[arm] + np.eye(self.d) * q_val
        
        # Innovation
        innovation = error
        
        # Kalman Gain
        # K = Sigma_prior * x^T * (x^T * Sigma_prior * x^T + R)^-1
        # Since x is a vector, x^T Sigma_prior x^T is a scalar
        sigma_x = context @ Sigma_prior @ context
        denom = sigma_x + self.R
        K = (Sigma_prior @ context) / denom
        
        # Update Theta
        self.theta[arm] = mu + K * innovation
        
        # Update Sigma (Joseph form for stability, or standard)
        # Standard: Sigma_post = (I - K x^T) Sigma_prior
        # Joseph: Sigma_post = (I - K x^T) Sigma_prior (I - K x^T)^T + K R K^T
        # For rank-1, standard is usually sufficient if K is well-conditioned
        I = np.eye(self.d)
        IK = I - np.outer(K, context)
        self.Sigma[arm] = IK @ Sigma_prior @ IK.T + np.outer(K, K) * self.R
        
        # Ensure symmetry and positive semi-definite
        self.Sigma[arm] = (self.Sigma[arm] + self.Sigma[arm].T) / 2.0
        
        self.t += 1
