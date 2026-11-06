"""Adaptive RLS with Multi-Scale Forgetting for Non-Stationary Linear Bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Policy using Recursive Least Squares (RLS) with Adaptive Multi-Scale Forgetting.
    
    Strategy:
    - Uses RLS for O(d^2) updates per step (faster than matrix inversion per step in some cases,
      and simpler than maintaining full covariance for Thompson Sampling).
    - Instead of a single fixed forgetting factor, we maintain a "trust" score for each arm.
    - If the prediction error is large, we aggressively reduce the effective forgetting factor
      (i.e., increase the rate of forgetting old data) to adapt quickly to regime shifts.
    - If the error is small, we slowly increase the forgetting factor (retain more history)
      to reduce variance in the estimate.
    - Selection is based on the estimated reward plus an exploration bonus proportional to
      the uncertainty (diagonal of the covariance matrix, which is tracked implicitly or explicitly).
      To keep it simple and fast, we track the covariance matrix P (inverse of A) and use
      an UCB-like score: x^T theta + alpha * sqrt(x^T P x).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize parameters for each arm
        # theta: weight vector (d,)
        # P: covariance matrix (d x d), initially large (uncertain)
        # A: precision matrix (d x d), initially small (confident? No, A = P^-1)
        # In RLS, we often update P directly.
        # Let's use the standard RLS update on P.
        # P_0 = (1/lambda_0) * I. A large P means high uncertainty.
        
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with a large value to encourage exploration
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Forgetting factor adaptation
        # lambda is the forgetting factor. 1.0 = no forgetting, 0.9 = fast forgetting.
        # We adapt this per arm.
        self.lambdas = [0.99 for _ in range(n_arms)]
        
        # Bounds for lambda
        self.lambda_min = 0.80  # Fastest forgetting
        self.lambda_max = 0.999 # Slowest forgetting (almost no forgetting)
        
        # Learning rate for adapting lambda
        self.lambda_lr = 0.05
        
        # Exploration parameter (alpha)
        # We can make this adaptive too, but let's start with a fixed one
        # or scale it with the uncertainty.
        self.alpha = 1.5
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback to greedy if running out of time
            scores = [np.dot(self.theta[a], context) for a in range(self.n)]
            return int(np.argmax(scores))

        x = context
        scores = []
        for a in range(self.n):
            # Predicted reward
            pred = np.dot(self.theta[a], x)
            
            # Uncertainty
            # var = x^T P x
            # We can compute this efficiently
            Px = self.P[a] @ x
            var = np.dot(x, Px)
            
            # UCB score
            score = pred + self.alpha * np.sqrt(var)
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        x = context
        theta = self.theta[arm]
        P = self.P[arm]
        lam = self.lambdas[arm]

        # 1. RLS Update
        # Standard RLS update:
        # K = P x / (lambda + x^T P x)
        # theta = theta + K * (y - x^T theta)
        # P = (P - K x^T P) / lambda
        
        Px = P @ x
        denom = lam + np.dot(x, Px)
        
        # Gain
        K = Px / denom
        
        # Prediction error
        e = reward - np.dot(theta, x)
        
        # Update theta
        self.theta[arm] = theta + K * e
        
        # Update P
        # P_new = (P - K x^T P) / lambda
        # To maintain symmetry and positive definiteness, we use:
        # P_new = (P - outer(K, x) @ P) / lambda
        # Or more stably:
        # P_new = (P - outer(K, x.T) @ P) / lambda
        # Let's use the form: P = (I - K x^T) P / lambda
        # (I - K x^T) P = P - K (x^T P) = P - K (Px)^T
        P_update = P - np.outer(K, Px)
        self.P[arm] = P_update / lam
        
        # Symmetrize to prevent numerical drift
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

        # 2. Adapt Forgetting Factor
        # If the error is large relative to the uncertainty, we might be in a new regime.
        # We should forget faster (lower lambda).
        # If the error is small, we are in a stable regime.
        # We should forget slower (higher lambda).
        
        # Normalize the error by the standard deviation of the prediction
        # std = sqrt(x^T P_old x + noise_var)
        # We don't explicitly track noise_var, but x^T P x is a good proxy for model uncertainty.
        # Let's use the residual e and the uncertainty var.
        var = np.dot(x, Px) # Note: Px was computed before update, so this is pre-update uncertainty
        std = np.sqrt(var + 1.0) # +1.0 to avoid division by zero and account for noise
        
        # Standardized error
        z = abs(e) / std
        
        # Target lambda:
        # If z is small (< 1), we are confident. Increase lambda towards max.
        # If z is large (> 2), we are surprised. Decrease lambda towards min.
        
        # Simple adaptation rule:
        # If z < 1.0: lambda += lr * (lambda_max - lambda) * (1 - z)
        # If z > 2.0: lambda -= lr * (lambda - lambda_min) * (z - 1)
        
        if z < 1.0:
            # Stable, increase lambda (retain more history)
            self.lambdas[arm] = lam + self.lambda_lr * (self.lambda_max - lam) * (1.0 - z)
        else:
            # Unstable, decrease lambda (forget faster)
            # Scale the decrease by how much z exceeds 1
            excess = z - 1.0
            # Cap the decrease to avoid jumping to min too fast
            decrease_factor = min(1.0, excess)
            self.lambdas[arm] = lam - self.lambda_lr * (lam - self.lambda_min) * decrease_factor
            
        # Clamp lambda
        self.lambdas[arm] = np.clip(self.lambdas[arm], self.lambda_min, self.lambda_max)

# EVOLVE-BLOCK-END
