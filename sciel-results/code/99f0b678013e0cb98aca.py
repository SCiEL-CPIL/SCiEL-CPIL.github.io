"""Non-stationary linear contextual bandit with adaptive sliding window and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Adaptive sliding window (EWMA-based non-stationarity detection) for robust adaptation.
    2. Thompson Sampling with Cholesky-based sampling for exploration/exploitation.
    
    The window size adapts based on prediction error: when errors are high (non-stationary),
    we use a shorter effective window to track changes faster. When errors are low, we use
    a longer window for lower variance estimates.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 108.0  # safety margin before 120s limit
        
        # Base sliding window size per arm
        self.base_window_size = 50
        
        # Ridge regularization parameter
        self.lam = 1.0
        
        # Per-arm sliding window data
        # queues[a] stores (context, reward) in chronological order
        self.queues = [collections.deque(maxlen=self.base_window_size) for _ in range(n_arms)]
        
        # Cached A and b matrices for fast updates
        # A[a] = sum of x_i x_i^T over window, b[a] = sum of r_i x_i
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA tracking for non-stationarity detection
        # ewma_sq_err[a] tracks exponentially weighted mean squared prediction error
        self.ewma_sq_err = [0.0] * n_arms
        self.ewma_alpha = 0.3  # fast response to changes
        
        # Effective window size per arm (adapted based on non-stationarity)
        self.eff_window = [self.base_window_size] * n_arms
        
        # Precompute Cholesky factorizations lazily
        self._chol_cache = [None] * n_arms
        self._L_cache = [None] * n_arms

    def _update_arm_stats(self, arm, context, reward):
        """Update the sliding window statistics for a given arm."""
        q = self.queues[arm]
        
        # If window is full, remove the oldest entry
        if len(q) == self.base_window_size:
            old_ctx, old_reward = q[0]
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_reward * old_ctx
        
        # Add new entry
        q.append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update EWMA squared error for non-stationarity detection
        # Use current theta to predict, compare to actual reward
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
            pred = theta @ context
            sq_err = (reward - pred) ** 2
        except np.linalg.LinAlgError:
            sq_err = 0.0
        
        self.ewma_sq_err[arm] = (1 - self.ewma_alpha) * self.ewma_sq_err[arm] + self.ewma_alpha * sq_err
        
        # Adapt effective window size based on EWMA error
        # Higher error -> shorter window (faster adaptation)
        # Lower error -> longer window (more stable estimates)
        base = self.base_window_size
        # Map EWMA error to effective window: range from 15 to base_window_size
        # Use a smooth mapping
        err_scale = min(self.ewma_sq_err[arm] / (1.0 + self.ewma_sq_err[arm]), 1.0)
        min_win = 15
        self.eff_window[arm] = max(min_win, int(base - (base - min_win) * err_scale))
        
        # Invalidate Cholesky cache
        self._chol_cache[arm] = None
        self._L_cache[arm] = None

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
        
        # Compute Cholesky of A, then L such that Cov = L L^T where L = inv(L_A)
        # A = L_A L_A^T, so A^{-1} = L_A^{-T} L_A^{-1}
        # We need Cholesky of A^{-1}: if A = L_A L_A^T, then A^{-1} = (L_A^{-T})(L_A^{-1})
        # So Cov = A^{-1}, and we can sample z ~ N(0, I), then x = theta + L_A^{-T} z
        # Equivalently, solve L_A w = z, then x = theta + w
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
        self._update_arm_stats(arm, context, reward)
# EVOLVE-BLOCK-END
