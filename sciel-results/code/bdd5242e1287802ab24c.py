"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Thompson Sampling for non-stationary linear bandits.
    
    Combines:
    - Thompson Sampling for exploration (sampling from posterior)
    - Adaptive forgetting based on Mahalanobis distance of residuals
    - Efficient matrix operations with deadline guard
    - Smooth decay modulation via EMA of normalized residuals
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_decay = 0.97       # faster base forgetting for non-stationarity
        self.shift_decay = 0.4       # strong forgetting on regime shift
        self.shift_threshold = 1.5   # trigger stronger forgetting earlier
        self.min_decay = 0.25        # allow more aggressive resetting
        self.t = 0
        
        # EMA tracking of normalized residuals for smooth decay
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.25        # faster EMA response to regime changes
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 148.0  # Leave 2s safety margin
        
        # Noise estimate for scaling
        self.noise_est = 1.0
        self.noise_alpha = 0.9

    def _safe(self):
        """Check if we're near the deadline."""
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

    def _compute_decay(self, arm, normalized_residual):
        """
        Compute smooth decay factor based on EMA of normalized residuals.
        Uses a smooth function to avoid discontinuities.
        """
        # Update EMA
        self.residual_ema[arm] = (1 - self.ema_alpha) * self.residual_ema[arm] + \
                                  self.ema_alpha * normalized_residual
        
        ema_val = self.residual_ema[arm]
        
        # Smooth transition: use logistic-like function
        # When ema_val < shift_threshold: close to base_decay
        # When ema_val > shift_threshold: close to shift_decay
        # Smooth transition width
        width = 1.0
        x = (ema_val - self.shift_threshold) / width
        # Sigmoid: 0.5 * (1 + tanh(x))
        sigmoid = 0.5 * (1 + np.tanh(x))
        
        # Interpolate between base and shift decay
        decay = self.base_decay + (self.shift_decay - self.base_decay) * sigmoid
        
        return max(decay, self.min_decay)

    def select(self, context):
        # Deadline guard: return greedy if near deadline
        if self._safe():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                A_inv = self._cholesky_inv(self.A[a])
                theta = A_inv @ self.b[a]
                mu = theta @ context
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm
        
        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            A_inv = self._cholesky_inv(self.A[a])
            theta_mean = A_inv @ self.b[a]
            mu = theta_mean @ context
            
            # Sample theta from N(theta_mean, A_inv * noise_var)
            # Use Cholesky of A_inv for sampling with ridge for stability
            try:
                ridge = 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(A_inv + ridge)
                z = np.random.randn(self.d)
                theta_sample = theta_mean + L @ z
                score = theta_sample @ context
            except np.linalg.LinAlgError:
                # Fallback to UCB if sampling fails
                var = context @ A_inv @ context
                std = np.sqrt(max(var, 1e-12))
                score = mu + 1.2 * std
            
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Compute current model prediction and uncertainty
        A_inv = self._cholesky_inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mu = theta @ context
        var = context @ A_inv @ context
        std = np.sqrt(max(var, 1e-12))
        
        # Normalized residual (Mahalanobis-like distance)
        if std > 1e-8:
            normalized_residual = abs(reward - mu) / std
        else:
            normalized_residual = abs(reward - mu)
        
        # Update noise estimate
        self.noise_est = self.noise_alpha * self.noise_est + \
                         (1 - self.noise_alpha) * (reward - mu) ** 2
        
        # Compute adaptive decay
        decay = self._compute_decay(arm, normalized_residual)
        
        # Apply forgetting with ridge regularization to maintain positive definiteness
        self.A[arm] = decay * self.A[arm]
        self.b[arm] = decay * self.b[arm]
        
        # Add new observation
        self.A[arm] = self.A[arm] + np.outer(context, context)
        self.b[arm] = self.b[arm] + reward * context
        
        # Ensure A remains well-conditioned: add small ridge if needed
        # (cheap check every few steps to avoid overhead)
        if self.t % 50 == 0:
            try:
                np.linalg.cholesky(self.A[arm])
            except np.linalg.LinAlgError:
                self.A[arm] += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
