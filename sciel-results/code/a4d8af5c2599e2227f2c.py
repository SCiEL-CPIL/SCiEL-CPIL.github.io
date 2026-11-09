"""Non-stationary linear contextual bandit with per-arm adaptive forgetting and Thompson Sampling."""
import numpy as np
import time


class Policy:
    """
    Adaptive per-arm forgetting LinTS (Linear Thompson Sampling).
    
    Key structural changes from seed:
    1. Thompson Sampling instead of UCB: samples theta from posterior for exploration.
    2. Per-arm adaptive forgetting factor based on reward variance (MAD-based robust scaling).
    3. Cholesky-based inverse maintenance for numerical stability and speed.
    4. Deadline guard to stay within time budget.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Per-arm state
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        # Base forgetting factor; will be modulated per-arm
        self.base_gamma = 0.97
        # Track reward history per arm for adaptive forgetting
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history = 50
        
        # Per-arm learned forgetting factors
        self.gammas = [self.base_gamma for _ in range(n_arms)]
        
        # Cholesky factors for efficient inverse
        self.L = [np.eye(dim) for _ in range(n_arms)]
        self.L_valid = [True for _ in range(n_arms)]

    def _maintain_cholesky(self, arm):
        """Maintain Cholesky factorization of A[arm]."""
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            self.L_valid[arm] = True
        except np.linalg.LinAlgError:
            # Add small jitter and retry
            self.A[arm] += 1e-6 * np.eye(self.d)
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                self.L_valid[arm] = True
            except np.linalg.LinAlgError:
                self.L_valid[arm] = False

    def _solve_cholesky(self, arm, rhs):
        """Solve A[arm] x = rhs using Cholesky factorization."""
        if not self.L_valid[arm]:
            return np.linalg.solve(self.A[arm], rhs)
        L = self.L[arm]
        # Solve L y = rhs, then L^T x = y
        y = np.linalg.solve(L, rhs)
        x = np.linalg.solve(L.T, y)
        return x

    def _update_forgetting(self, arm, reward):
        """Adaptively adjust forgetting factor based on reward variability."""
        hist = self.reward_history[arm]
        hist.append(reward)
        if len(hist) > self.max_history:
            hist.pop(0)
        
        if len(hist) >= 5:
            # Use MAD (median absolute deviation) for robust scale estimate
            med = np.median(hist)
            mad = np.median(np.abs(np.array(hist) - med))
            # Convert MAD to std-like estimate
            std_est = 1.4826 * mad if mad > 0 else 0.01
            
            # Higher variance -> faster forgetting (lower gamma)
            # Scale: if std is large, reduce gamma to forget faster
            # gamma in [0.9, 0.995]
            base_std = 0.5  # reference scale
            ratio = std_est / base_std
            # Clamp ratio
            ratio = np.clip(ratio, 0.1, 10.0)
            # Adaptive gamma: higher ratio -> lower gamma (faster forgetting)
            new_gamma = self.base_gamma - 0.05 * np.log1p(ratio)
            new_gamma = np.clip(new_gamma, 0.90, 0.995)
            
            # Smooth the forgetting factor update
            self.gammas[arm] = 0.95 * self.gammas[arm] + 0.05 * new_gamma

    def select(self, context):
        """Select arm using Thompson Sampling."""
        if self.t > 0 and (time.time() - self.start_time) > self.deadline:
            # Deadline exceeded; return arm with highest mean prediction
            return self._greedy_select(context)
        
        context = np.asarray(context, dtype=np.float64)
        samples = []
        for a in range(self.n):
            # Compute theta estimate
            theta = self._solve_cholesky(a, self.b[a])
            
            # Compute posterior covariance: (A^{-1}) = (L^{-1})^T L^{-1}
            if self.L_valid[a]:
                # Covariance = inv(A) = inv(L^T L) = (L^{-1})^T (L^{-1})
                # We need to sample from N(theta, inv(A))
                # Sample: theta + inv(A) @ z where z ~ N(0, I)
                # inv(A) @ z = (L^{-1})^T (L^{-1}) z
                # Let w = L^{-1} z, then inv(A) z = (L^{-1})^T w
                z = np.random.standard_normal(self.d)
                w = np.linalg.solve(self.L[a], z)
                cov_sample = np.linalg.solve(self.L[a].T, w)
                theta_sample = theta + cov_sample
            else:
                # Fallback: use direct inverse
                inv_A = np.linalg.inv(self.A[a])
                theta_sample = theta + np.random.multivariate_normal(np.zeros(self.d), inv_A)
            
            samples.append(theta_sample @ context)
        
        self.t += 1
        return int(np.argmax(samples))

    def _greedy_select(self, context):
        """Fallback greedy selection if deadline exceeded."""
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self._solve_cholesky(a, self.b[a])
            scores.append(theta @ context)
        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update model with observed reward."""
        context = np.asarray(context, dtype=np.float64)
        gamma = self.gammas[arm]
        
        # Apply forgetting
        self.A[arm] = gamma * self.A[arm]
        self.b[arm] = gamma * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Maintain Cholesky factorization
        self._maintain_cholesky(arm)
        
        # Adapt forgetting factor
        self._update_forgetting(arm, reward)
        
        # Small regularization to keep A well-conditioned
        self.A[arm] += 1e-4 * np.eye(self.d)
        if self.L_valid[arm]:
            self._maintain_cholesky(arm)
