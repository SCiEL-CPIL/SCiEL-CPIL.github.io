"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Online Bayesian Linear Regression 
    with Exponential Forgetting (Recursive Least Squares).
    
    Strategy:
    1. Maintain per-arm posterior parameters (mean theta, covariance P) for linear reward model.
    2. Use exponential forgetting factor gamma < 1 to decay old observations' influence.
    3. This allows the model to adapt to changing arm weights (non-stationarity) 
       without the high computational cost of sliding window regression.
    4. Use Thompson Sampling for exploration: sample theta from posterior N(theta, P).
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Forgetting factor: closer to 1 means longer memory, closer to 0 means faster adaptation
        # 0.95 is a standard choice for moderate non-stationarity
        self.gamma = 0.95
        
        # Prior covariance: large initial uncertainty
        self.prior_cov = 10.0 * np.eye(dim)
        
        # Initialize per-arm posteriors
        # theta: mean of the weight vector
        # P: covariance of the weight vector
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [self.prior_cov.copy() for _ in range(n_arms)]
        
        # Noise variance (assumed constant for simplicity, can be tuned)
        self.noise_var = 1.0
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit
        self.elapsed = 0.0

    def _update_posterior(self, arm, context, reward):
        """
        Update the posterior for a specific arm using Recursive Least Squares 
        with exponential forgetting.
        """
        # Retrieve current parameters
        theta = self.theta[arm]
        P = self.P[arm]
        
        # Exponential forgetting: scale old information by gamma
        # This effectively down-weights older data
        P = P * self.gamma
        # Note: theta is the mean. The "information" is in P^{-1}.
        # The update rule for RLS with forgetting is:
        # K = P x / (noise_var + x^T P x)
        # P_new = (P - K x^T P) / gamma  (Wait, standard RLS with forgetting gamma is:
        # P_t = (1/gamma) * (I - K_t x_t^T) P_{t-1} ? No.
        
        # Let's use the standard Kalman Filter update with process noise or forgetting.
        # A common approach for non-stationary linear bandits is:
        # 1. Decay covariance: P = P / gamma (increases uncertainty)
        # 2. Standard Kalman update
        
        # Step 1: Decay (Increase uncertainty to forget)
        P = P / self.gamma
        
        # Step 2: Kalman Update
        # Innovation covariance: S = noise_var + x^T P x
        x = context
        S = self.noise_var + x @ P @ x
        
        # Kalman Gain: K = P x / S
        K = (P @ x) / S
        
        # Update mean: theta = theta + K * (reward - x^T theta)
        innovation = reward - x @ theta
        theta = theta + K * innovation
        
        # Update covariance: P = (I - K x^T) P
        # Efficient update: P = P - (P x x^T P) / S
        Px = P @ x
        P = P - (np.outer(Px, Px) / S)
        
        # Ensure symmetry and positive semi-definiteness (numerical stability)
        P = (P + P.T) / 2.0
        
        # Store back
        self.theta[arm] = theta
        self.P[arm] = P

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: greedy based on current mean
            context = np.array(context)
            means = [np.dot(self.theta[a], context) for a in range(self.n_arms)]
            return int(np.argmax(means))

        context = np.array(context)
        scores = []
        
        for a in range(self.n_arms):
            theta = self.theta[a]
            P = self.P[a]
            
            # Mean score
            mean_score = np.dot(theta, context)
            
            # Variance of the score
            # Var(context^T theta) = context^T P context
            var_score = context @ P @ context
            std_score = np.sqrt(max(var_score, 1e-8))
            
            # Thompson Sampling: Sample score from N(mean_score, std_score)
            sampled_score = np.random.normal(mean_score, std_score)
            scores.append(sampled_score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.array(context)
        self._update_posterior(arm, context, reward)
        self.elapsed = time.time() - self.start_time
# EVOLVE-BLOCK-END
