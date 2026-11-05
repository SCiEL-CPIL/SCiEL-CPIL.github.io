import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Kalman Filter Parameters
        # Initial state covariance P (uncertainty)
        # High initial uncertainty encourages exploration
        self.P = [np.eye(self.d) * 10.0 for _ in range(self.n)]
        
        # State estimates theta
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        
        # Measurement noise R
        # Assuming standard Gaussian rewards, we estimate this.
        # A small R means we trust the measurement more.
        self.R = 0.1
        
        # Base process noise Q
        # This controls how much we expect the weights to change per step.
        # We will scale this adaptively.
        self.base_Q = 0.01
        
        # UCB parameter
        self.alpha = 1.0
        
        # Adaptive noise tracking
        # We keep a short history of squared innovations per arm to scale Q
        self.innovation_hist = [np.zeros(50) for _ in range(self.n)]
        self.hist_idx = [0 for _ in range(self.n)]
        
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Predicted mean
            mu = np.dot(self.theta[a], context_vec)
            
            # Uncertainty term for UCB
            # In Kalman Filter, P is the covariance of the state estimate.
            # The variance of the prediction x^T theta is x^T P x.
            px = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, px)
            
            if var < 0:
                var = 0.0
                
            std = np.sqrt(var)
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # 1. Calculate Innovation (Prediction Error)
        pred = np.dot(self.theta[arm], context_vec)
        innovation = reward - pred
        
        # 2. Update Innovation History for Adaptive Q
        self.innovation_hist[arm][self.hist_idx[arm]] = innovation ** 2
        self.hist_idx[arm] = (self.hist_idx[arm] + 1) % 50
        
        # Calculate average recent innovation magnitude
        # If errors are large, the environment is changing rapidly -> increase Q
        # If errors are small, environment is stable -> decrease Q (towards base_Q)
        recent_innovations = self.innovation_hist[arm]
        avg_innov = np.mean(recent_innovations)
        
        # Scale Q based on recent performance
        # If avg_innov is much larger than R, the model is lagging.
        # We boost Q to allow faster tracking.
        # Cap the boost to prevent instability.
        noise_scale = 1.0 + np.clip(avg_innov / (self.R + 1e-6), 0.0, 10.0)
        Q = self.base_Q * noise_scale * np.eye(self.d)
        
        # 3. Kalman Predict Step
        # State prediction: theta_pred = theta (random walk model, no drift)
        # Covariance prediction: P_pred = P + Q
        P_pred = self.P[arm] + Q
        
        # 4. Kalman Update Step
        # Innovation covariance: S = H P_pred H^T + R
        # Here H = x^T (1 x d), so S = x^T P_pred x + R (scalar)
        P_pred_x = np.dot(P_pred, context_vec)
        S = np.dot(context_vec, P_pred_x) + self.R
        
        # Avoid division by zero
        if S < 1e-12:
            S = 1e-12
            
        # Kalman Gain: K = P_pred H^T / S
        # K = P_pred x / S (d x 1 vector)
        K = P_pred_x / S
        
        # State Update: theta = theta_pred + K * innovation
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update: P = (I - K H) P_pred
        # I - K x^T is (d x d)
        # We can compute this as P - K (x^T P_pred)
        # Note: P_pred x is already computed as P_pred_x
        self.P[arm] = P_pred - np.outer(K, P_pred_x)
        
        # Numerical stability: Symmetrize P
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        # Ensure positive semi-definiteness by clipping eigenvalues if needed?
        # For speed, we just symmetrize. If P becomes indefinite, var might be negative, 
        # which we handle in select().
        
        self.t += 1
