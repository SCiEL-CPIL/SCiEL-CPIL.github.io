"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Kernelized Linear Bandit with RBF kernel and exponential forgetting."""

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.ridge_lambda = 1.0
        self.gamma = 0.98  # Forgetting factor
        self.rbf_sigma = 1.0  # RBF kernel bandwidth
        self.min_history = 3
        self.max_history = 50  # Cap on stored points per arm

        # Per-arm storage: list of (context, reward, weight) where weight decays
        self.histories = [[] for _ in range(self.n_arms)]
        
        # Track elapsed time for deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0

        # Cache for kernel matrix computations
        self._kernel_cache = {}

    def _rbf_kernel(self, x1, x2):
        """Compute RBF kernel between two points."""
        diff = x1 - x2
        return np.exp(-np.dot(diff, diff) / (2.0 * self.rbf_sigma ** 2))

    def _compute_kernel_matrix(self, contexts):
        """Compute kernel matrix for a list of contexts."""
        n = len(contexts)
        if n == 0:
            return np.zeros((0, 0))
        
        # Check cache
        key = tuple(tuple(c) for c in contexts)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._rbf_kernel(contexts[i], contexts[j])
                K[j, i] = K[i, j]
        
        # Cache result (limit cache size)
        if len(self._kernel_cache) > 100:
            self._kernel_cache.clear()
        self._kernel_cache[key] = K
        
        return K

    def _get_arm_stats(self, arm, context):
        """Compute posterior mean and variance for given arm and context."""
        hist = self.histories[arm]
        
        if len(hist) < self.min_history:
            return 0.0, 1.0 / self.ridge_lambda

        # Extract contexts, rewards, and weights
        contexts = [h[0] for h in hist]
        rewards = np.array([h[1] for h in hist])
        weights = np.array([h[2] for h in hist])

        n = len(contexts)
        
        # Compute kernel matrix K
        K = self._compute_kernel_matrix(contexts)
        
        # Weighted kernel matrix: K_w = K * outer(sqrt(w), sqrt(w))
        sqrt_w = np.sqrt(weights)
        K_w = K * np.outer(sqrt_w, sqrt_w)
        
        # Add regularization
        A = K_w + self.ridge_lambda * np.eye(n)
        
        # Solve for alpha: alpha = A^{-1} (K_w @ y)
        try:
            Ky = K_w @ rewards
            alpha = np.linalg.solve(A, Ky)
        except np.linalg.LinAlgError:
            return 0.0, 1.0 / self.ridge_lambda

        # Compute kernel vector k between new context and stored contexts
        k = np.array([self._rbf_kernel(context, ctx) for ctx in contexts])
        
        # Posterior mean: mu = k^T alpha
        mean = np.dot(k, alpha)
        
        # Posterior variance: var = k(context, context) - k^T A^{-1} k
        try:
            A_inv_k = np.linalg.solve(A, k)
            var = self._rbf_kernel(context, context) - np.dot(k, A_inv_k)
            var = max(var, 1e-6)
        except np.linalg.LinAlgError:
            var = 1.0 / self.ridge_lambda

        return mean, var

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(hash(tuple(context)) % self.n_arms)

        samples = []
        for a in range(self.n_arms):
            mean, var = self._get_arm_stats(a, context)
            sample = np.random.normal(mean, np.sqrt(var))
            samples.append(sample)

        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        # Apply forgetting to existing weights
        for i in range(len(self.histories[arm])):
            self.histories[arm][i] = (self.histories[arm][i][0], 
                                      self.histories[arm][i][1], 
                                      self.histories[arm][i][2] * self.gamma)
        
        # Add new observation with weight 1.0
        self.histories[arm].append((context, reward, 1.0))
        
        # Trim history if too large
        if len(self.histories[arm]) > self.max_history:
            # Keep only the most recent points with significant weight
            self.histories[arm] = [h for h in self.histories[arm] if h[2] > 1e-4]
            if len(self.histories[arm]) > self.max_history:
                self.histories[arm] = self.histories[arm][-self.max_history:]
        
        # Clear kernel cache when history changes significantly
        if len(self.histories[arm]) % 10 == 0:
            self._kernel_cache.clear()
# EVOLVE-BLOCK-END
