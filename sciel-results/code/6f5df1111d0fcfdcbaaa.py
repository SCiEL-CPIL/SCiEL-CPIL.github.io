import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Strategy: Sliding-Window RLS with UCB Exploration
        # Instead of Thompson Sampling (stochastic), we use UCB (deterministic + confidence).
        # Instead of Bayesian Posterior (mu, Sigma), we use RLS (w, P) with exponential forgetting.
        # RLS is computationally lighter (no Cholesky sampling per arm per step) and 
        # the trace(P) provides a natural measure of uncertainty for UCB.
        
        # Forgetting factor: Controls the effective window size.
        # gamma = 0.95 is a reasonable start for non-stationary environments.
        self.gamma = 0.95
        
        # RLS Parameters per arm:
        # P[a]: Inverse covariance matrix (dim, dim)
        # w[a]: Weight vector (dim,)
        
        # Initialize P with large identity to encourage initial exploration (high uncertainty)
        # Initialize w to zeros
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        self.w = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB Exploration Parameter
        # We scale the confidence term by sqrt(lambda * trace(P * x x^T P)) or similar.
        # A common form is c * sqrt(x^T P x).
        # We need to tune c. Let's start with a value that balances exploitation/exploration.
        self.alpha = 1.0 # Exploration factor
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy using current weights
            context_vec = np.array(context)
            scores = [self.w[a] @ context_vec for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        # Calculate scores for each arm
        # Score = w_a^T x + alpha * sqrt(x^T P_a x)
        # x^T P_a x is the variance of the prediction for this context under arm a's model.
        
        scores = []
        for a in range(self.n):
            mean_score = self.w[a] @ context_vec
            
            # Calculate variance term: x^T P x
            # P is symmetric positive definite (usually)
            # Use np.dot for efficiency
            Px = self.P[a] @ context_vec
            variance_term = context_vec @ Px
            
            # Ensure non-negative due to numerical errors
            if variance_term < 0:
                variance_term = 0.0
                
            confidence_bonus = self.alpha * np.sqrt(variance_term)
            
            scores.append(mean_score + confidence_bonus)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context_vec = np.array(context)
        
        # RLS Update with Forgetting Factor
        # P_new = (gamma * P_old^{-1} + x x^T)^{-1}
        # This is equivalent to:
        # K = P_old x / (gamma + x^T P_old x)  (Gain vector)
        # w_new = w_old + K * (reward - x^T w_old)
        # P_new = (P_old - K x^T) / gamma
        
        P_old = self.P[arm]
        w_old = self.w[arm]
        
        # Calculate denominator: gamma + x^T P_old x
        Px_old = P_old @ context_vec
        denom = self.gamma + context_vec @ Px_old
        
        # Avoid division by zero (should not happen with gamma > 0 and P positive definite)
        if denom < 1e-12:
            return
            
        # Calculate Gain K
        K = Px_old / denom
        
        # Calculate Prediction Error
        pred = w_old @ context_vec
        error = reward - pred
        
        # Update weights
        w_new = w_old + K * error
        
        # Update inverse covariance
        # P_new = (P_old - np.outer(K, context_vec)) / gamma
        P_new = (P_old - np.outer(K, context_vec)) / self.gamma
        
        # Symmetrize P_new to prevent numerical drift
        # P_new = (P_new + P_new.T) / 2
        # This is optional but good for stability. However, RLS update is inherently symmetric if P_old is.
        # Let's just enforce symmetry to be safe.
        P_new = (P_new + P_new.T) * 0.5
        
        # Store updated parameters
        self.P[arm] = P_new
        self.w[arm] = w_new
