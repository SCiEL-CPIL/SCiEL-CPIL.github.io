import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Extended dimension: original context + one-hot arm indicators
        self.ext_dim = dim + n_arms
        self.ridge = 1.0
        
        # Sliding window size for non-stationarity
        self.window_size = 80
        
        # Global model state
        # A is (ext_dim, ext_dim), b is (ext_dim,)
        self.A = np.eye(self.ext_dim) * self.ridge
        self.b = np.zeros(self.ext_dim)
        self.inv_A = np.eye(self.ext_dim) / self.ridge
        
        # History for sliding window
        self.history = deque(maxlen=self.window_size)
        
        # Thompson Sampling noise
        self.noise_var = 1.0
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0
        
        # Random state
        self.rng = np.random.default_rng()
        
        # Pre-compute context parts for speed if possible? No, context varies.
        # But we can pre-compute the arm one-hot vectors
        self.arm_vectors = [np.zeros(self.ext_dim) for _ in range(n_arms)]
        for i in range(n_arms):
            self.arm_vectors[i][dim + i] = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _smw_add(self, x):
        """Add x x^T to A and update inverse using Sherman-Morrison-Woodbury."""
        A_inv = self.inv_A
        tmp = A_inv @ x
        denom = 1.0 + x @ tmp
        if abs(denom) < 1e-10:
            # Fallback
            self.A = self.A + np.outer(x, x)
            try:
                self.inv_A = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self.inv_A = np.eye(self.ext_dim)
        else:
            self.inv_A = A_inv - np.outer(tmp, tmp) / denom
            self.A = self.A + np.outer(x, x)

    def _smw_subtract(self, x):
        """Subtract x x^T from A and update inverse."""
        A_inv = self.inv_A
        tmp = A_inv @ x
        denom = 1.0 - x @ tmp
        if abs(denom) < 1e-10 or denom <= 0:
            # Fallback: recompute
            self.A = self.A - np.outer(x, x)
            try:
                self.inv_A = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self.inv_A = np.eye(self.ext_dim)
        else:
            self.inv_A = A_inv + np.outer(tmp, tmp) / denom
            self.A = self.A - np.outer(x, x)

    def _build_feature(self, context, arm):
        """Construct the extended feature vector: [context, one_hot_arm]"""
        # context is shape (d,), arm is int
        # We can use np.concatenate or pre-allocated buffer
        # For speed, let's use a small helper
        # Since d is small, concatenation is fine
        return np.concatenate([context, self.arm_vectors[arm]])

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick best mean
            scores = []
            for a in range(self.n):
                x = self._build_feature(context, a)
                theta = self.inv_A @ self.b
                scores.append(context @ theta[:self.d] + theta[self.d + a]) # Wait, simpler: x @ theta
            # Actually, just compute full x @ theta
            theta = self.inv_A @ self.b
            scores = []
            for a in range(self.n):
                x = self._build_feature(context, a)
                scores.append(x @ theta)
            return int(np.argmax(scores))

        # Thompson Sampling
        theta_hat = self.inv_A @ self.b
        
        # Sample theta
        try:
            L = np.linalg.cholesky(self.inv_A)
            z = self.rng.normal(size=self.ext_dim)
            theta_sample = theta_hat + self.noise_var * (L @ z)
        except np.linalg.LinAlgError:
            theta_sample = theta_hat

        scores = []
        for a in range(self.n):
            x = self._build_feature(context, a)
            scores.append(x @ theta_sample)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        # Evict if window is full
        if len(self.history) == self.window_size:
            old_ctx, old_arm, old_reward = self.history[0]
            old_x = self._build_feature(old_ctx, old_arm)
            self._smw_subtract(old_x)
            self.b = self.b - old_reward * old_x
            
        # Add new
        new_x = self._build_feature(context, arm)
        self.history.append((context, arm, reward))
        self._smw_add(new_x)
        self.b = self.b + reward * new_x
