"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary Linear Contextual Bandit using a Global Sliding Window Linear Model.
    
    Strategy:
    1. Instead of maintaining separate models per arm, we use a single global linear model.
    2. The context vector is augmented with a one-hot encoding of the arm index.
       This allows the model to capture interactions between context and specific arms 
       while sharing statistical strength across arms where possible.
    3. We use a sliding window of the most recent N observations to handle non-stationarity.
       This effectively "forgets" old data that may no longer be relevant.
    4. We use Thompson Sampling: we maintain the posterior distribution of the weights 
       based on the sliding window data and sample from it to select the next arm.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        # Total dimensionality: original context dim + n_arms one-hot features
        self.total_dim = dim + n_arms
        
        # Sliding window size
        # Needs to be large enough to estimate weights but small enough to adapt
        self.window_size = 200
        self.window = deque(maxlen=self.window_size)
        
        # Prior for Thompson Sampling
        # Large prior variance to encourage initial exploration
        self.prior_cov = 10.0 * np.eye(self.total_dim)
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit
        
        # Counter for how many times we've updated
        self.t = 0

    def _augment_context(self, context, arm):
        """
        Augment the context vector with a one-hot encoding of the arm.
        """
        # Ensure context is a numpy array
        context = np.asarray(context, dtype=np.float64)
        
        # Create one-hot vector for the arm
        one_hot = np.zeros(self.n_arms, dtype=np.float64)
        one_hot[arm] = 1.0
        
        # Concatenate
        return np.concatenate([context, one_hot])

    def _compute_posterior(self, X, y):
        """
        Compute the posterior mean and covariance for the linear model y = X^T theta + noise
        using the data in X (N x D) and y (N,).
        
        We assume a Gaussian prior theta ~ N(0, prior_cov) and Gaussian noise ~ N(0, sigma^2).
        Posterior:
        theta_post ~ N(mu_post, Sigma_post)
        Sigma_post = (prior_cov^{-1} + X^T X / sigma^2)^{-1}
        mu_post = Sigma_post * (X^T y / sigma^2)
        
        For computational efficiency and numerical stability with sliding windows,
        we can use the Sherman-Morrison-Woodbury formula or just solve the linear system
        if the window is small. Given window_size=200 and dim likely small (<10),
        direct inversion or solve is feasible.
        
        However, doing this every step is O(N * D^3). 
        To be safer on time, we can update recursively or just use a simpler heuristic 
        if N is large. But 200 is small.
        
        Let's use the standard Bayesian linear regression update.
        """
        N = X.shape[0]
        D = X.shape[1]
        
        if N == 0:
            return np.zeros(D), self.prior_cov.copy()
        
        sigma2 = 1.0
        
        # Regularization to ensure invertibility
        # prior_cov^{-1} is 1/10 * I
        A = self.prior_cov + (X.T @ X) / sigma2
        b = (X.T @ y) / sigma2
        
        # Solve A theta = b
        try:
            # Use solve for better numerical stability than inv
            mu_post = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            mu_post = np.linalg.pinv(A) @ b
            
        Sigma_post = np.linalg.inv(A)
        
        # Ensure symmetry
        Sigma_post = (Sigma_post + Sigma_post.T) / 2.0
        
        return mu_post, Sigma_post

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: Greedy with current best estimate
            # We need a fast estimate. If we haven't computed the posterior recently, 
            # we might have stale data. For the guard, let's just pick a random arm 
            # or the one with highest historical average reward if available.
            # To be safe and fast, return 0.
            return 0

        context = np.asarray(context, dtype=np.float64)
        
        # If we have no data, explore randomly
        if len(self.window) < self.n_arms:
            return np.random.randint(self.n_arms)
        
        # Compute posterior from the sliding window
        # Construct X and y from the window
        X_list = []
        y_list = []
        for ctx, arm, rew in self.window:
            X_list.append(self._augment_context(ctx, arm))
            y_list.append(rew)
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        mu_post, Sigma_post = self._compute_posterior(X, y)
        
        # Thompson Sampling
        # Sample theta from N(mu_post, Sigma_post)
        theta_sample = np.random.multivariate_normal(mu_post, Sigma_post)
        
        # Evaluate all arms
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            x_a = self._augment_context(context, a)
            score = np.dot(theta_sample, x_a)
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.window.append((context, arm, reward))
        self.t += 1
# EVOLVE-BLOCK-END
