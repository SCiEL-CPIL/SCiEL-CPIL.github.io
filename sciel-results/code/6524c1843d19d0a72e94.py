import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # State variables
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with larger covariance to encourage exploration
        # 10.0 is a good balance between trust in prior (0) and data
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Adaptive Q parameters
        # Faster decay to adapt quickly to non-stationarity
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.ema_decay = 0.85 
        self.q_base = 0.1
        self.q_max = 1.0
        self.q_scale = 2.0
        
        # Exploration parameter
        # Start high, decay slowly to a non-zero floor to track drift
        self.beta_init = 3.0
        self.beta_min = 1.0
        self.beta_decay = 0.999
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0 # Seconds

    def _get_adaptive_q(self, arm_idx):
        # Calculate Q based on current error EMA
        # We want Q to increase when errors are high to allow theta to change
        # tanh is smooth, but maybe a simple clip is faster and more predictable
        # q_val = self.q_base + (self.q_max - self.q_base) * np.tanh(self.q_scale * self.error_ema[arm_idx])
        
        # Alternative: Linear scaling with clipping
        # raw_q = self.q_base + self.q_scale * self.error_ema[arm_idx]
        # q_val = min(raw_q, self.q_max)
        
        # Let's stick to tanh but tune parameters
        q_val = self.q_base + (self.q_max - self.q_base) * np.tanh(self.q_scale * self.error_ema[arm_idx])
        return np.eye(self.d) * q_val

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Fallback: just return the arm with highest current theta prediction
            context = np.asarray(context)
            preds = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(preds))

        context = np.asarray(context)
        scores = []
        
        # Dynamic beta: decays slowly to a minimum
        self.beta = self.beta_min + (self.beta_init - self.beta_min) * (self.beta_decay ** self.t)
        
        for a in range(self.n):
            # Prediction
            pred = self.theta[a] @ context
            
            # Uncertainty
            # P is symmetric positive definite
            var = context @ self.P[a] @ context
            # Numerical stability: ensure variance is non-negative
            std = np.sqrt(max(var, 1e-6))
            
            # UCB Score
            score = pred + self.beta * std
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context)
        
        # 1. Calculate prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # 2. Update Error EMA
        self.error_ema[arm] = self.ema_decay * self.error_ema[arm] + (1 - self.ema_decay) * (error ** 2)
        
        # 3. Calculate Adaptive Q
        Q = self._get_adaptive_q(arm)
        
        # 4. Prediction Step: P_pred = P + Q
        P_pred = self.P[arm] + Q
        
        # 5. Update Step (Kalman)
        # Assume observation noise sigma^2 = 1.0
        sigma2 = 1.0
        
        # Compute Kalman Gain
        # K = P_pred x^T (x^T P_pred x + sigma^2)^-1
        Px = P_pred @ context
        denom = context @ Px + sigma2
        K = Px / denom
        
        # Update Theta
        self.theta[arm] = self.theta[arm] + K * error
        
        # Update P
        # Use Joseph form for numerical stability
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K sigma^2 K^T
        I = np.eye(self.d)
        KxT = np.outer(K, context)
        IKxT = I - KxT
        
        # Optimization: Instead of full matrix multiplication, we can use
        # P_new = P_pred - P_pred @ x @ x^T @ P_pred / denom
        # This is the standard Kalman update formula which is cheaper and stable enough for small d
        # P_new = P_pred - np.outer(Px, Px) / denom
        
        # Let's use the cheaper formula as it's standard and stable for this scale
        self.P[arm] = P_pred - np.outer(Px, Px) / denom
        
        # Ensure symmetry
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
