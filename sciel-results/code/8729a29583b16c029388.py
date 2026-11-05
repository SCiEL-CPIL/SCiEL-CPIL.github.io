"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy using RLS and UCB.
    
    Strategy:
    - Per-arm weight estimation using Recursive Least Squares (RLS) with a 
      forgetting factor to handle non-stationarity.
    - Maintains full covariance matrices to estimate prediction uncertainty.
    - UCB (Upper Confidence Bound) exploration to balance exploitation and 
      exploration based on model uncertainty.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize weight vectors to zero
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initialize inverse covariance matrices. 
        # Large initial value (low uncertainty) to encourage early exploration.
        self.Ps = [np.eye(dim) * 1000.0 for _ in range(n_arms)]
        
        # Hyperparameters
        # Forgetting factor: close to 1 means slow forgetting, but we want to 
        # adapt to changes. 0.95-0.99 is typical for non-stationary RLS.
        self.lambda_rls = 0.95 
        # UCB parameter: controls the exploration-exploitation trade-off.
        # Higher alpha means more exploration.
        self.alpha_ucb = 1.5
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin before 150s limit

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))
        
        # Ensure context is a numpy array for efficient computation
        context = np.asarray(context)
        
        # Compute UCB scores for each arm
        ucb_scores = np.zeros(self.n)
        for a in range(self.n):
            # Predicted mean reward
            pred_mean = self.thetas[a] @ context
            
            # Prediction variance: context^T P context
            # P is the inverse covariance matrix of the weights
            pred_var = context @ self.Ps[a] @ context
            
            # UCB = mean + alpha * sqrt(variance)
            ucb_scores[a] = pred_mean + self.alpha_ucb * np.sqrt(pred_var)
        
        return int(np.argmax(ucb_scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
            
        context = np.asarray(context)
        
        # RLS Update Step
        # 1. Compute gain vector
        # K = P * x / (lambda + x^T * P * x)
        Px = self.Ps[arm] @ context
        denom = self.lambda_rls + context @ Px
        K = Px / denom
        
        # 2. Update weight vector
        # error = y - x^T * theta
        pred = self.thetas[arm] @ context
        error = reward - pred
        self.thetas[arm] += K * error
        
        # 3. Update inverse covariance matrix
        # P_new = (P - K * x^T * P) / lambda
        # Using the identity: P_new = (1/lambda) * (P - P * x * x^T * P / (lambda + x^T * P * x))
        # This is numerically stable and efficient.
        self.Ps[arm] = (self.Ps[arm] - np.outer(Px, Px) / denom) / self.lambda_rls
# EVOLVE-BLOCK-END
