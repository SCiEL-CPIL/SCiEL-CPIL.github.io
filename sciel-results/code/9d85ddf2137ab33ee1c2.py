"""Non-Stationary Linear Bandit with Multi-Scale Ensemble and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Strategy:
    - Multi-Scale Ensemble: Instead of a single adaptive forgetting factor, we maintain
      multiple RLS models per arm with fixed, geometrically spaced forgetting factors.
      This allows the policy to automatically track changes at different timescales.
    - Selection: For each arm, we compute a weighted average of the predictions from
      the ensemble. The weights are based on the recent prediction errors (inverse error weighting).
    - Exploration: We use a UCB-like bonus based on the variance of the ensemble predictions
      (or the covariance of the best-performing model in the ensemble). To keep it simple
      and robust, we use the covariance of the model with the lowest recent error.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Define a set of forgetting factors (geometric progression)
        # Fast forgetting (0.8) to slow forgetting (0.99)
        self.lambdas = [0.8, 0.85, 0.90, 0.95, 0.99]
        self.n_models = len(self.lambdas)
        
        # Initialize parameters for each arm and each model in the ensemble
        # theta: (n_arms, n_models, dim)
        self.theta = np.zeros((n_arms, self.n_models, dim))
        
        # P: covariance matrices (n_arms, n_models, dim, dim)
        # Initialize with high uncertainty
        self.P = np.zeros((n_arms, self.n_models, dim, dim))
        for a in range(n_arms):
            for m in range(self.n_models):
                self.P[a, m] = np.eye(dim) * 100.0
        
        # Track recent prediction errors for each arm and model to weight the ensemble
        # We keep a small buffer of recent errors to compute a robust weight
        self.error_buffer_size = 10
        self.errors = np.zeros((n_arms, self.n_models, self.error_buffer_size))
        self.error_idx = 0  # Circular buffer index
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback to greedy using the first model
            scores = [np.dot(self.theta[a, 0], context) for a in range(self.n)]
            return int(np.argmax(scores))

        x = context
        scores = []
        for a in range(self.n):
            # Compute weighted average prediction and uncertainty
            preds = np.zeros(self.n_models)
            vars = np.zeros(self.n_models)
            weights = np.zeros(self.n_models)
            
            for m in range(self.n_models):
                # Prediction
                preds[m] = np.dot(self.theta[a, m], x)
                
                # Uncertainty (variance)
                Px = self.P[a, m] @ x
                vars[m] = np.dot(x, Px)
                
                # Weight based on recent errors (inverse MSE)
                # Add a small epsilon to avoid division by zero
                mse = np.mean(self.errors[a, m] ** 2) + 1e-6
                weights[m] = 1.0 / mse
                
            # Normalize weights
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights = weights / sum_weights
            else:
                weights = np.ones(self.n_models) / self.n_models
            
            # Weighted prediction
            pred = np.dot(weights, preds)
            
            # Weighted variance (approximation: weighted sum of variances)
            # Note: This ignores covariance between models, which is a reasonable approximation
            var = np.dot(weights, vars)
            
            # UCB score
            score = pred + self.alpha * np.sqrt(var)
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        x = context
        
        # Update all models in the ensemble for this arm
        for m in range(self.n_models):
            theta = self.theta[arm, m]
            P = self.P[arm, m]
            lam = self.lambdas[m]

            # RLS Update
            Px = P @ x
            denom = lam + np.dot(x, Px)
            
            # Gain
            K = Px / denom
            
            # Prediction error
            e = reward - np.dot(theta, x)
            
            # Update theta
            self.theta[arm, m] = theta + K * e
            
            # Update P
            P_update = P - np.outer(K, Px)
            self.P[arm, m] = P_update / lam
            
            # Symmetrize
            self.P[arm, m] = (self.P[arm, m] + self.P[arm, m].T) / 2.0
            
            # Store error in circular buffer
            self.errors[arm, m, self.error_idx] = e

        # Advance circular buffer index
        self.error_idx = (self.error_idx + 1) % self.error_buffer_size

# EVOLVE-BLOCK-END
