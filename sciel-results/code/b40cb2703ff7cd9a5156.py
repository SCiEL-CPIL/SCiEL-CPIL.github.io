"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with sliding-window ridge regression for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 150s kill

        # Sliding window size for non-stationarity
        self.window_size = 50
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]

        # Ridge regularization
        self.lambda_reg = 1.0

        # Pre-allocated arrays for speed
        self.A_cache = [None] * n_arms
        self.b_cache = [None] * n_arms
        self.L_cache = [None] * n_arms
        self.L_valid = [False] * n_arms

        # Per-arm visit counts
        self.visits = [0] * n_arms

        # Reward noise estimate
        self.noise_est = [1.0] * n_arms
        self.noise_window = 20

    def _rebuild_model(self, arm):
        """Rebuild ridge regression model from sliding window data."""
        ctxs = self.contexts[arm]
        rews = self.rewards[arm]
        
        if len(ctxs) < 2:
            self.A_cache[arm] = np.eye(self.d) * self.lambda_reg
            self.b_cache[arm] = np.zeros(self.d)
            self.L_cache[arm] = np.eye(self.d) * np.sqrt(self.lambda_reg)
            self.L_valid[arm] = True
            return
        
        X = np.array(ctxs)
        y = np.array(rews)
        
        # Ridge regression: A = X^T X + lambda*I, b = X^T y
        self.A_cache[arm] = X.T @ X + self.lambda_reg * np.eye(self.d)
        self.b_cache[arm] = X.T @ y
        
        # Cholesky decomposition
        try:
            self.L_cache[arm] = np.linalg.cholesky(self.A_cache[arm])
            self.L_valid[arm] = True
        except np.linalg.LinAlgError:
            # Add regularization if Cholesky fails
            self.A_cache[arm] += 1e-6 * np.eye(self.d)
            try:
                self.L_cache[arm] = np.linalg.cholesky(self.A_cache[arm])
                self.L_valid[arm] = True
            except np.linalg.LinAlgError:
                self.L_valid[arm] = False
    
    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            return 0

        scores = []
        for a in range(self.n):
            if not self.L_valid[a]:
                # Fallback: uniform random
                scores.append(np.random.randn())
                continue
            
            L = self.L_cache[a]
            A = self.A_cache[a]
            b = self.b_cache[a]
            
            # Solve for theta: A theta = b
            try:
                theta = np.linalg.solve(L.T, np.linalg.solve(L, b))
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Compute variance: context^T A^{-1} context
            try:
                v = np.linalg.solve(L, context)
                var = np.dot(v, v)
            except np.linalg.LinAlgError:
                inv = np.linalg.inv(A)
                var = context @ inv @ context
            
            # Thompson Sampling: sample from posterior
            # Posterior: theta ~ N(theta_hat, sigma^2 * A^{-1})
            # Sample: theta_hat + sigma * A^{-1/2} * z where z ~ N(0, I)
            noise = self.noise_est[a]
            if noise < 0.1:
                noise = 0.1  # Minimum noise to avoid degenerate samples
            
            # Sample using Cholesky: A^{-1/2} = L^{-1} L^{-T}
            # So A^{-1/2} z = L^{-1} (L^{-T} z)
            z = np.random.randn(self.d)
            try:
                # L^{-T} z: solve L^T w = z
                w = np.linalg.solve(L.T, z)
                # L^{-1} w: solve L theta_sample_offset = w
                offset = np.linalg.solve(L, w)
            except np.linalg.LinAlgError:
                # Fallback: use variance-based scaling
                offset = np.random.randn(self.d) * np.sqrt(max(var, 0)) / np.sqrt(self.d)
            
            theta_sample = theta + noise * offset
            scores.append(theta_sample @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1

        # Update visit count
        self.visits[arm] += 1

        # Update sliding window
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
        
        if len(self.contexts[arm]) > self.window_size:
            self.contexts[arm].pop(0)
            self.rewards[arm].pop(0)
        
        # Update noise estimate using recent residuals
        if len(self.rewards[arm]) >= 3:
            recent_rews = self.rewards[arm][-self.noise_window:]
            if len(recent_rews) >= 2:
                mean_r = np.mean(recent_rews)
                var_r = np.var(recent_rews)
                # Estimate noise as std of rewards (rough upper bound)
                self.noise_est[arm] = max(np.sqrt(var_r), 0.1)
        
        # Rebuild model from sliding window
        self._rebuild_model(arm)
# EVOLVE-BLOCK-END
