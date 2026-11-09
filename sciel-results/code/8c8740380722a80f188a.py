"""Non-stationary linear contextual bandit with global model and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Global linear model with context-augmented features (one-hot arm + context).
    2. Sliding window for robust adaptation to changing arm weights.
    3. Thompson Sampling for exploration/exploitation.
    
    The feature vector for arm a and context x is [e_a, x] where e_a is a one-hot
    vector. This captures both common trends and arm-specific offsets in a unified
    parameter vector, reducing the effective exploration space.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # safety margin before 120s limit
        
        # Augmented feature dimension: n_arms (one-hot) + dim (context)
        self.d_aug = n_arms + dim
        
        # Sliding window size
        self.window_size = 50
        
        # Ridge regularization parameter
        self.lam = 1.0
        
        # Precompute one-hot vectors for each arm
        self.one_hot = np.zeros((n_arms, n_arms))
        for a in range(n_arms):
            self.one_hot[a, a] = 1.0
        
        # Sliding window data
        self.queues = collections.deque(maxlen=self.window_size)
        
        # Global A and b matrices
        self.A = self.lam * np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        
        # Cholesky cache
        self._chol_cache = None

    def _augment(self, arm, context):
        """Create augmented feature vector [one_hot(arm), context]."""
        return np.concatenate([self.one_hot[arm], context])

    def _update_stats(self, arm, context, reward):
        """Update the sliding window statistics."""
        if len(self.queues) == self.window_size:
            old_arm, old_ctx, old_reward = self.queues[0]
            old_feat = self._augment(old_arm, old_ctx)
            self.A -= np.outer(old_feat, old_feat)
            self.b -= old_reward * old_feat
        
        feat = self._augment(arm, context)
        self.queues.append((arm, context, reward))
        self.A += np.outer(feat, feat)
        self.b += reward * feat
        
        # Invalidate Cholesky cache
        self._chol_cache = None

    def _get_theta_and_cov(self):
        """
        Compute the posterior mean (theta) and covariance.
        theta = A^{-1} b, Cov = A^{-1}
        """
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d_aug)
        
        try:
            cov = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            cov = np.eye(self.d_aug)
        
        return theta, cov

    def select(self, context):
        """
        Select an arm using Thompson Sampling with the global model.
        Sample a parameter vector from the posterior, compute predicted rewards
        for all arms using their augmented features, and select the best.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check: if running out of time, use greedy fallback
        if time.time() > self.deadline:
            theta, _ = self._get_theta_and_cov()
            scores = np.array([theta @ self._augment(a, context) for a in range(self.n)])
            return int(np.argmax(scores))
        
        # Thompson Sampling
        theta, cov = self._get_theta_and_cov()
        
        # Sample from N(theta, cov)
        sampled_theta = theta + np.random.multivariate_normal(np.zeros(self.d_aug), cov)
        
        # Compute predicted reward for each arm
        scores = np.array([sampled_theta @ self._augment(a, context) for a in range(self.n)])
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the policy with a new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        self._update_stats(arm, context, reward)
# EVOLVE-BLOCK-END
