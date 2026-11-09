"""
Thompson Sampling with Sliding Window for Non-Stationary Linear Bandits.
Uses per-arm sliding windows to estimate the posterior distribution of weights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        
        # Parameters
        self.window_size = 60  # Sliding window size
        self.reg = 1.0         # Regularization parameter
        self.noise_floor = 0.1 # Minimum noise variance to avoid zero variance
        
        # Data structures
        # Deques to store (context, reward) for each arm
        self.windows = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Cache for statistics to avoid recomputation if possible, 
        # but with sliding window, we recompute on select.
        # We will compute stats on the fly in select.
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin
        
        # RNG for Thompson Sampling
        self.rng = np.random.default_rng(42)

    def _get_posterior_stats(self, arm):
        """
        Compute mean and covariance of the posterior for the given arm
        based on the current sliding window.
        Returns:
            theta_hat: posterior mean (dim,)
            cov: posterior covariance (dim, dim)
            sigma_sq: estimated noise variance
        """
        W = len(self.windows[arm])
        d = self.dim
        
        if W == 0:
            # Prior: N(0, I)
            return np.zeros(d), np.eye(d), 1.0

        # Extract data
        # For small W and d, this is fast
        data = list(self.windows[arm])
        X = np.array([x for x, r in data])
        r = np.array([r for x, r in data])
        
        # Compute sufficient statistics
        # S_xx = X^T X + reg * I
        # S_xr = X^T r
        S_xx = X.T @ X + self.reg * np.eye(d)
        S_xr = X.T @ r
        
        # Invert S_xx
        try:
            S_xx_inv = np.linalg.inv(S_xx)
        except np.linalg.LinAlgError:
            # Fallback to regularized inverse if singular
            S_xx_inv = np.linalg.pinv(S_xx)
            
        theta_hat = S_xx_inv @ S_xr
        cov = S_xx_inv
        
        # Estimate noise variance
        # Residuals
        preds = X @ theta_hat
        residuals = r - preds
        sigma_sq = np.mean(residuals ** 2)
        
        # Ensure sigma_sq is not too small
        if sigma_sq < self.noise_floor:
            sigma_sq = self.noise_floor
            
        return theta_hat, cov, sigma_sq

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback: greedy based on last known means or random
            # To be safe, pick arm with highest mean estimate quickly
            # Or just pick arm 0 if we are in trouble
            # Better: use cached means if available, else 0
            # For simplicity in emergency, pick 0
            return 0

        context = np.asarray(context)
        
        scores = []
        for a in range(self.n_arms):
            theta_hat, cov, sigma_sq = self._get_posterior_stats(a)
            
            # Sample theta from N(theta_hat, sigma_sq * cov)
            # Use Cholesky decomposition for efficient sampling if possible
            # cov is symmetric positive semi-definite
            try:
                L = np.linalg.cholesky(cov)
                z = self.rng.standard_normal(self.dim)
                theta_sample = theta_hat + sigma_sq * (L @ z)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use eigendecomposition or just mean
                # Or add small jitter
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.maximum(eigvals, 0)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    z = self.rng.standard_normal(self.dim)
                    theta_sample = theta_hat + sigma_sq * (L @ z)
                except:
                    theta_sample = theta_hat
            
            score = context @ theta_sample
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.windows[arm].append((np.asarray(context), reward))
        self.t += 1
# EVOLVE-BLOCK-END
