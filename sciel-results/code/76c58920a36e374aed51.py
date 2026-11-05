"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Thompson Sampling for non-stationary linear bandits.
    
    Combines:
    - Thompson Sampling for exploration (sampling from posterior)
    - Adaptive forgetting based on normalized prediction residuals
    - Efficient matrix inversion via Cholesky with fallback
    - Time-aware safety margin
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_decay = 0.99       # mild forgetting each step
        self.shift_decay = 0.5       # strong forgetting on regime shift
        self.shift_threshold = 2.0   # residual/uncertainty threshold
        self.min_decay = 0.3         # floor for decay factor
        self.t = 0
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Leave 5s margin
        
        # Track recent errors for adaptive beta
        self.recent_errors = []
        self.error_window = 50
        
        # Noise estimate
        self.noise_est = 1.0
        self.noise_alpha = 0.9

    def _safe(self):
        return time.time() >= self.deadline

    def _cholesky_inv(self, M):
        """Stable inverse via Cholesky, with fallback to pinv."""
        try:
            L = np.linalg.cholesky(M)
            inv_L = np.linalg.inv(L)
            inv_M = inv_L.T @ inv_L
            return inv_M
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def _compute_adaptive_decay(self, normalized_residual):
        """
        Compute decay factor based on prediction error.
        High normalized residual -> strong forgetting (lower decay).
        Low normalized residual -> mild forgetting (higher decay).
        """
        if normalized_residual > self.shift_threshold:
            # Strong shift: use shift_decay or even lower if very large error
            excess = (normalized_residual - self.shift_threshold) / self.shift_threshold
            decay = self.shift_decay - 0.2 * min(excess, 1.0)
            return max(decay, self.min_decay)
        else:
            # Mild forgetting, slightly modulated by residual
            # If residual is close to threshold, lean toward stronger forgetting
            proximity = normalized_residual / self.shift_threshold
            decay = self.base_decay - 0.02 * proximity
            return max(decay, self.min_decay)

    def select(self, context):
        context = np.asarray(context, dtype=float)
        
        if self._safe():
            # Fallback: return greedy arm based on current theta
            scores = []
            for a in range(self.n):
                A_inv = self._cholesky_inv(self.A[a])
                theta = A_inv @ self.b[a]
                mu = theta @ context
                scores.append(mu)
            return int(np.argmax(scores))
        
        # Thompson Sampling: sample from posterior for each arm
        samples = []
        for a in range(self.n):
            A_inv = self._cholesky_inv(self.A[a])
            theta_mean = A_inv @ self.b[a]
            mu = theta_mean @ context
            
            # Sample theta ~ N(theta_mean, A_inv)
            # Use Cholesky of A_inv for sampling
            try:
                L_inv = np.linalg.cholesky(A_inv)
                z = np.random.randn(self.d)
                theta_sample = theta_mean + L_inv @ z
                sample_score = theta_sample @ context
            except np.linalg.LinAlgError:
                # Fallback to UCB-style if sampling fails
                var = context @ A_inv @ context
                std = np.sqrt(max(var, 1e-12))
                sample_score = mu + 1.0 * std
            
            samples.append(sample_score)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self.t += 1
        
        if self._safe():
            # Minimal update to maintain state
            self.A[arm] = self.A[arm] + np.outer(context, context)
            self.b[arm] = self.b[arm] + reward * context
            return
        
        # Compute current model prediction and uncertainty for this arm
        A_inv = self._cholesky_inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mu = theta @ context
        var = context @ A_inv @ context
        std = np.sqrt(max(var, 1e-12))
        
        # Update noise estimate
        error = reward - mu
        self.noise_est = self.noise_alpha * self.noise_est + (1 - self.noise_alpha) * (error ** 2)
        noise_std = np.sqrt(max(self.noise_est, 1e-8))
        
        # Normalized residual: how surprised are we?
        if noise_std > 1e-8:
            normalized_residual = abs(reward - mu) / noise_std
        else:
            normalized_residual = abs(reward - mu)
        
        # Track recent errors for potential future adaptation
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)
        
        # Decide decay factor based on normalized residual
        decay = self._compute_adaptive_decay(normalized_residual)
        
        # Apply forgetting
        self.A[arm] = decay * self.A[arm]
        self.b[arm] = decay * self.b[arm]
        
        # Add new observation
        self.A[arm] = self.A[arm] + np.outer(context, context)
        self.b[arm] = self.b[arm] + reward * context
# EVOLVE-BLOCK-END
