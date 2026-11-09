"""Non-stationary linear contextual bandit with adaptive discounting and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Adaptive discounted linear regression with non-stationarity detection.
    2. Thompson Sampling with Cholesky-based sampling for exploration/exploitation.
    
    The discount factor adapts based on prediction error: when errors are high (non-stationary),
    we use a smaller gamma (faster forgetting) to track changes. When errors are low, we use
    a larger gamma for more stable estimates.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 108.0  # safety margin before 120s limit
        
        # Adaptive discount factor per arm
        # Start with moderate discount, adapt based on non-stationarity
        self.gamma = [0.98] * n_arms
        self.gamma_min = 0.85
        self.gamma_max = 0.995
        
        # Ridge regularization parameter
        self.lam = 1.0
        
        # Per-arm discounted statistics
        # A[a] = sum of gamma^t * x_i x_i^T, b[a] = sum of gamma^t * r_i x_i
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA tracking for non-stationarity detection
        # ewma_sq_err[a] tracks exponentially weighted mean squared prediction error
        self.ewma_sq_err = [0.0] * n_arms
        self.ewma_alpha = 0.15  # responsive to drift
        
        # Threshold for detecting non-stationarity
        self.err_threshold = 1.5
        
        # Precompute Cholesky factorizations lazily
        self._L_cache = [None] * n_arms

    def _adapt_gamma(self, arm):
        """Adapt the discount factor based on prediction error."""
        err = self.ewma_sq_err[arm]
        # If error is high, decrease gamma (faster forgetting)
        # If error is low, increase gamma (slower forgetting, more stable)
        if err > self.err_threshold:
            # Non-stationary: decrease gamma
            self.gamma[arm] = max(self.gamma_min, self.gamma[arm] - 0.01)
        else:
            # Stationary: increase gamma towards max
            self.gamma[arm] = min(self.gamma_max, self.gamma[arm] + 0.005)

    def _get_theta_and_L(self, arm):
        """
        Compute the posterior mean (theta) and Cholesky factor of covariance for an arm.
        Uses ridge regression: theta = A^{-1} b, Cov = A^{-1}
        where A includes the ridge term.
        """
        A = self.A[arm]
        b = self.b[arm]
        
        # Compute theta = A^{-1} b
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
        
        # Compute Cholesky of A for sampling
        try:
            L_A = np.linalg.cholesky(A)
            self._L_cache[arm] = L_A
        except np.linalg.LinAlgError:
            L_A = np.eye(self.d)
            self._L_cache[arm] = L_A
        
        return theta, L_A

    def select(self, context):
        """
        Select an arm using Thompson Sampling with Cholesky-based sampling.
        For each arm, sample a parameter vector from the posterior,
        compute the predicted reward, and select the arm with the highest sample.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check: if we're running out of time, use a simple greedy fallback
        if time.time() > self.deadline:
            # Greedy fallback: pick arm with highest predicted reward (no exploration)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta, _ = self._get_theta_and_L(a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)
        
        # Thompson Sampling with Cholesky-based sampling
        samples = []
        for a in range(self.n):
            theta, L_A = self._get_theta_and_L(a)
            # Sample z ~ N(0, I), then solve L_A w = z, sampled_theta = theta + w
            z = np.random.randn(self.d)
            try:
                w = np.linalg.solve(L_A, z)
            except np.linalg.LinAlgError:
                w = z
            sampled_theta = theta + w
            # Predicted reward for this arm
            pred_reward = sampled_theta @ context
            samples.append(pred_reward)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """
        Update the policy with a new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Adapt gamma based on current error
        self._adapt_gamma(arm)
        gamma = self.gamma[arm]
        
        # Update discounted statistics
        # A = gamma * A + x x^T, b = gamma * b + r * x
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Update EWMA squared error for non-stationarity detection
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta @ context
            sq_err = (reward - pred) ** 2
        except np.linalg.LinAlgError:
            sq_err = 0.0
        
        self.ewma_sq_err[arm] = (1 - self.ewma_alpha) * self.ewma_sq_err[arm] + self.ewma_alpha * sq_err
        
        # Invalidate Cholesky cache
        self._L_cache[arm] = None
# EVOLVE-BLOCK-END
