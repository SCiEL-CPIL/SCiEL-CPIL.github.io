import numpy as np
from collections import deque

class Policy:
    """Thompson Sampling with Sliding Window for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 50  # Sliding window size
        self.lambda_reg = 1.0  # Ridge regression regularization
        
        # Store recent (context, reward) pairs for each arm
        self.buffers = [deque(maxlen=self.window_size) for _ in range(n_arms)]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        sampled_thetas = []
        
        for a in range(self.n):
            # Get data for arm a
            if len(self.buffers[a]) == 0:
                # No data, use prior (zero mean, high variance)
                theta_hat = np.zeros(self.d)
                # Use a large variance for exploration when no data
                cov = 10.0 * np.eye(self.d)
            else:
                # Extract contexts and rewards
                data = list(self.buffers[a])
                X = np.array([d[0] for d in data])
                y = np.array([d[1] for d in data])
                
                # Ridge Regression: (X^T X + lambda I)^-1 X^T y
                XtX = X.T @ X
                XtX_reg = XtX + self.lambda_reg * np.eye(self.d)
                Xty = X.T @ y
                
                try:
                    # Solve for theta_hat and Covariance
                    # Cov = inv(XtX_reg)
                    cov = np.linalg.inv(XtX_reg)
                    theta_hat = cov @ Xty
                except np.linalg.LinAlgError:
                    # Fallback if singular (shouldn't happen with ridge)
                    cov = 10.0 * np.eye(self.d)
                    theta_hat = np.zeros(self.d)
            
            # Sample from N(theta_hat, cov)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            sampled_thetas.append(theta_sample)
        
        # Predict rewards for the current context using sampled thetas
        rewards = np.array([np.dot(context, theta) for theta in sampled_thetas])
        
        return int(np.argmax(rewards))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.buffers[arm].append((context, reward))
