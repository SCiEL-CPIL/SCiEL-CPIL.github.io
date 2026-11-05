# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Per-Arm Robust RLS with Huber loss.
    
    Structural change from Kalman filter:
    - Uses recursive least squares with exponential forgetting instead of KF
    - Huber loss downweights large innovations (outliers/regime shifts)
    - Sliding-window inverse covariance reset when residual exceeds dynamic threshold
    - Simpler per-arm independent models (no trend decomposition)
    - Tanh-saturated adaptive forgetting factor based on recent error
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Per-arm RLS state
        # theta: weight vector estimate
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # P: inverse of accumulated Hessian (covariance-like), initialized large
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Forgetting factor (lambda): close to 1 means slow forgetting
        # Adaptive: starts at 0.95, increases toward 0.99 as errors stabilize
        self.lambda_base = 0.95
        self.lambda_max = 0.99
        self.forgetting = [0.95 for _ in range(n_arms)]
        
        # Huber loss parameter: delta controls transition from quadratic to linear
        self.huber_delta = 2.0
        
        # Exploration coefficient
        self.alpha = 1.0
        
        # Counts for each arm
        self.counts = [0 for _ in range(n_arms)]
        
        # Sliding window of recent innovations for adaptive threshold
        self.recent_innovations = [[] for _ in range(n_arms)]
        self.window_size = 30
        
        # Adaptive forgetting via tanh-saturated EMA of error
        self.ema_errors = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1  # EMA smoothing factor

    def _huber_weight(self, innovation, delta):
        """Compute Huber loss weight for an innovation.
        
        For |y| <= delta: weight = 1 (quadratic regime, full update)
        For |y| > delta: weight = delta / |y| (linear regime, downweighted)
        """
        abs_y = abs(innovation)
        if abs_y <= delta:
            return 1.0
        else:
            return delta / abs_y

    def _should_reset(self, arm, innovation):
        """Check if we should reset the inverse covariance for this arm.
        
        Reset triggers when |innovation| exceeds dynamic threshold derived
        from current P matrix and recent error history.
        """
        x_norm = 1.0  # approximate; will use actual context norm in update
        # Dynamic threshold: sqrt of P-based variance scaled by recent error scale
        P_diag_mean = np.mean(np.diag(self.P[arm]))
        dynamic_threshold = 3.0 * np.sqrt(max(P_diag_mean, 0.01))
        
        # Also check against recent innovations
        if len(self.recent_innovations[arm]) > 5:
            recent_std = np.std(self.recent_innovations[arm])
            if recent_std > 0.1:
                dynamic_threshold = max(dynamic_threshold, 3.0 * recent_std)
        
        return abs(innovation) > dynamic_threshold

    def select(self, context):
        """Select arm using UCB with RLS estimates."""
        self.t += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            # Predicted reward
            mu = self.theta[a] @ context
            
            # Uncertainty from RLS covariance
            Px = self.P[a] @ context
            var = context @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Adaptive exploration: reduce as we learn more about this arm
            count_factor = 1.0 / (1.0 + np.sqrt(self.counts[a]))
            exploration = self.alpha * std * count_factor
            
            scores.append(mu + exploration)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update RLS for the selected arm with Huber loss and adaptive forgetting."""
        self.counts[arm] += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return
        
        a = arm
        x = context
        r = reward
        
        # Compute innovation (prediction error)
        y = r - self.theta[a] @ x
        
        # Update sliding window of innovations
        self.recent_innovations[a].append(y)
        if len(self.recent_innovations[a]) > self.window_size:
            self.recent_innovations[a].pop(0)
        
        # Check if we should reset (regime change detection)
        if self._should_reset(a, y):
            # Reset P to encourage re-exploration, keep theta
            self.P[a] = np.eye(self.d) * 50.0
            # Clear recent innovations
            self.recent_innovations[a] = []
            # Reset EMA error
            self.ema_errors[a] = 0.0
            return
        
        # Compute Huber weight for robust update
        w = self._huber_weight(y, self.huber_delta)
        
        # Adaptive forgetting factor using tanh-saturated EMA
        # Large errors -> lower lambda (faster forgetting)
        # Small errors -> higher lambda (slower forgetting)
        self.ema_errors[a] = (1 - self.ema_alpha) * self.ema_errors[a] + self.ema_alpha * abs(y)
        
        # Map EMA error to forgetting factor using tanh saturation
        # ema_error ~ 0 -> lambda near lambda_max
        # ema_error ~ 5+ -> lambda near lambda_base
        tanh_val = np.tanh(self.ema_errors[a] / 3.0)  # saturates around 5
        self.forgetting[a] = self.lambda_max - tanh_val * (self.lambda_max - self.lambda_base)
        
        lam = self.forgetting[a]
        
        # RLS update with Huber weighting
        # P_pred = P / lambda
        P_pred = self.P[a] / lam
        
        # Scaled innovation with Huber weight
        y_scaled = w * y
        
        # Innovation covariance: S = x^T P_pred x + 1 (measurement noise)
        S = x @ P_pred @ x + 1.0
        
        # Kalman-like gain
        K = P_pred @ x / S
        
        # Update theta
        self.theta[a] = self.theta[a] + K * y_scaled
        
        # Update P (Joseph form for stability)
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T
        
        # Ensure symmetry
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        
        # Clamp P to prevent numerical issues
        np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 1e-6))

# EVOLVE-BLOCK-END
