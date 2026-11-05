# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Thompson Sampling Linear Bandit with Kalman Filter Updates.
    
    Structural change: Replaces deterministic UCB with Thompson Sampling (TS).
    The policy maintains per-arm posterior Gaussian beliefs over weight vectors.
    The mean is updated via a Kalman filter step, which naturally handles the
    linear model structure. The covariance is inflated by an adaptive process
    noise term (Q) derived from the standardized innovation magnitude, allowing
    the policy to automatically increase exploration when the environment
    becomes non-stationary or when prediction errors are large.
    
    This combines the robust state-space modeling of the Kalman filter (from inspirations)
    with the probabilistic exploration of Thompson Sampling (from the directive).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage initial exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive Process Noise (Q) parameters
        # Base Q is small, but can grow up to Q_max if innovations are large
        self.Q_base = 3e-4
        self.Q_max = 0.04
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12  # EMA smoothing factor

        # Observation Noise (R)
        # Lower R allows for more aggressive updates, assuming rewards are relatively low noise
        self.obs_noise = 0.1

        # Reset mechanism for catastrophic drift
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0  # Safety margin before 150s kill

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self.Q_base
        
        ema_val = self._innov_ema[arm]
        # Map EMA to [0, 1] range using tanh to ensure bounded Q
        scaled = np.tanh(ema_val / 0.4)
        q = self.Q_base + (self.Q_max - self.Q_base) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance to encourage re-exploration."""
        # Threshold for 'large' innovation (squared)
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft reset: inflate covariance to encourage re-exploration
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check: if close to deadline, exploit best estimate (greedy)
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            
            # Uncertainty for Thompson Sampling
            # Sample from N(mean, variance)
            var = context @ self.P[a] @ context
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Thompson Sampling: sample from the posterior
            # We sample the score directly for efficiency
            score = np.random.normal(mean, std)
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward
        
        # Deadline guard: if we are running out of time, skip heavy updates
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict (Process Noise)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)
        
        # 2. Calculate Innovation
        y = r - self.theta[a] @ x
        
        # 3. Update Step (Kalman)
        # K = P_pred x (x^T P_pred x + sigma^2)^-1
        sigma2 = self.obs_noise
        Px = P_pred @ x
        denom = x @ Px + sigma2
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom
        
        # Update Theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P (Joseph Form)
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * sigma2
        
        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        
        # Track prediction error for adaptive Q
        # Use standardized innovation magnitude
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq
        
        # Check for reset
        self._maybe_reset(a, innov_sq)
        
        self.count[a] += 1
# EVOLVE-BLOCK-END
