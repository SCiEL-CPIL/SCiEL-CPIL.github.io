"""Non-stationary linear contextual bandit with RBF kernelization and exponential forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Kernelized Linear Bandit with RBF kernel and exponential forgetting.
    Uses dual formulation for efficient handling of non-stationarity.
    Combines Thompson Sampling for exploration with kernel methods for non-linearity.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 150s kill

        # RBF kernel parameters
        self.gamma = 1.0  # 1 / (2 * sigma^2)
        self.kernel_type = 'rbf'

        # Exponential forgetting rate (gamma_forget = exp(-lambda))
        self.forgetting_rate = 0.95  # Half-life ~13 steps
        
        # Ridge regularization
        self.lambda_reg = 1.0

        # Dual variables per arm: list of (context, reward, weight)
        # We store recent data points with their exponential weights
        self.data = [[] for _ in range(n_arms)]  # each entry: (context, reward)
        self.weights = [[] for _ in range(n_arms)]  # corresponding exponential weights
        
        # Cache kernel matrices for speed
        self.K_cache = [None] * n_arms
        self.k_cache = [None] * n_arms  # kernel vector for current context
        self.valid = [False] * n_arms

        # Maximum number of data points to keep (memory bound)
        self.max_data = 100

        # Noise estimate for Thompson Sampling
        self.noise_est = [1.0] * n_arms

        # Precompute identity for ridge
        self.I = np.eye(self.d)

    def _rbf_kernel(self, x1, x2):
        """Compute RBF kernel between two points."""
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))

    def _rbf_kernel_matrix(self, X1, X2):
        """Compute RBF kernel matrix between two sets of points."""
        # X1: (n1, d), X2: (n2, d)
        # Using broadcasting for efficiency
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        # Clip to avoid negative values due to numerical errors
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-self.gamma * sq_dists)

    def _rebuild_model(self, arm):
        """Rebuild kernel ridge regression model for the given arm."""
        data = self.data[arm]
        weights = self.weights[arm]
        
        if len(data) == 0:
            self.K_cache[arm] = None
            self.valid[arm] = False
            return
        
        X = np.array(data)
        y = np.array(weights) * np.array([r for _, r in data])
        w = np.array(weights)
        
        n = len(X)
        
        # Compute kernel matrix K (n x n)
        K = self._rbf_kernel_matrix(X, X)
        
        # Weighted kernel ridge regression
        # K_w = K * sqrt(w_i) * sqrt(w_j) element-wise
        sqrt_w = np.sqrt(w)
        K_w = K * np.outer(sqrt_w, sqrt_w)
        
        # Add regularization
        K_w += self.lambda_reg * np.eye(n)
        
        # Store for later use
        self.K_cache[arm] = K_w
        self.valid[arm] = True
        
        # Precompute Cholesky factorization
        try:
            self.L_cache = np.linalg.cholesky(K_w)
            self.L_valid = True
        except np.linalg.LinAlgError:
            K_w += 1e-6 * np.eye(n)
            try:
                self.L_cache = np.linalg.cholesky(K_w)
                self.L_valid = True
            except np.linalg.LinAlgError:
                self.L_valid = False

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            return 0

        context = np.array(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            if not self.valid[a] or self.K_cache[a] is None:
                # No data yet: use random exploration
                scores.append(np.random.randn())
                continue
            
            data = self.data[a]
            weights = self.weights[a]
            X = np.array(data)
            n = len(X)
            
            # Compute kernel vector k(x, X)
            k = self._rbf_kernel_matrix(np.array([context]), X)[0]
            
            # Compute mean prediction: k^T (K + lambda*I)^{-1} y
            # where y = w * rewards
            y_weighted = np.array(weights) * np.array([r for _, r in data])
            
            # Solve for alpha: (K_w) alpha = y_weighted
            # where K_w = K * outer(sqrt_w, sqrt_w) + lambda*I
            try:
                if hasattr(self, 'L_cache') and self.L_valid:
                    # Use Cholesky factorization
                    alpha = np.linalg.solve(self.L_cache.T, np.linalg.solve(self.L_cache, y_weighted))
                else:
                    alpha = np.linalg.solve(self.K_cache[a], y_weighted)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(self.K_cache[a], y_weighted, rcond=None)[0]
            
            mean_pred = np.dot(k, alpha)
            
            # Compute variance for Thompson Sampling
            # Var = k(x,x) - k^T (K + lambda*I)^{-1} k
            k_xx = 1.0  # RBF kernel at same point
            try:
                if hasattr(self, 'L_cache') and self.L_valid:
                    v = np.linalg.solve(self.L_cache.T, np.linalg.solve(self.L_cache, k))
                else:
                    v = np.linalg.solve(self.K_cache[a], k)
                var = max(k_xx - np.dot(k, v), 0.0)
            except np.linalg.LinAlgError:
                var = 1.0  # Fallback
            
            # Thompson Sampling: sample from posterior
            noise = self.noise_est[a]
            if noise < 0.05:
                noise = 0.05
            
            sample = mean_pred + noise * np.sqrt(var) * np.random.randn()
            scores.append(sample)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.array(context, dtype=np.float64)
        
        # Update exponential weights for all existing data points
        for i in range(len(self.weights[arm])):
            self.weights[arm][i] *= self.forgetting_rate
        
        # Add new data point with weight 1
        self.data[arm].append((context, reward))
        self.weights[arm].append(1.0)
        
        # Trim if too many data points
        if len(self.data[arm]) > self.max_data:
            # Remove oldest data point
            self.data[arm].pop(0)
            self.weights[arm].pop(0)
        
        # Update noise estimate
        if len(self.data[arm]) >= 3:
            recent = self.data[arm][-20:]
            rews = [r for _, r in recent]
            if len(rews) >= 2:
                self.noise_est[arm] = max(np.std(rews), 0.05)
        
        # Rebuild model
        self._rebuild_model(arm)
# EVOLVE-BLOCK-END
