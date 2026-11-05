import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with adaptive process noise, UCB exploration, and occasional resets.
    
    Key improvements over baseline:
    - Adaptive process noise via innovation EMA with tanh saturation.
    - Joseph-form covariance update for numerical stability.
    - Count-scaled UCB bonus that decays as we learn more about each arm.
    - Global alpha decay to reduce exploration over time.
    - Periodic soft-reset of covariance when prediction errors spike,
      allowing faster recovery from large weight shifts.
    - Tighter deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0  # safety margin before 120s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize covariance large to encourage exploration early
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 3e-4
        self._Q_max = 0.04
        # Context-scaled process noise coefficient
        self._lambda_ctx = 5e-4
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12  # EMA smoothing factor
        # Track context norm for context-scaled Q
        self._ctx_norm_sq = [0.0 for _ in range(n_arms)]
        self._ctx_norm_alpha = 0.2  # EMA smoothing for context norm

        # Observation noise
        self._R = 0.1

        # UCB parameter (adaptive)
        self.alpha = 1.2
        self._alpha_min = 0.3
        self._alpha_decay_rate = 500.0

        # Reset mechanism: track consecutive large innovations
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3.0  # number of consecutive large innovations
        self._reset_cov_scale = 5.0  # scale to reset covariance to

        # RBF-based context warping for non-linear structure capture
        # Per-arm RBF centers and weights
        self._n_rbf = 5
        self._rbf_centers = [np.random.randn(self._n_rbf, dim) * 0.5 for _ in range(n_arms)]
        self._rbf_weights = [np.zeros(self._n_rbf) for _ in range(n_arms)]
        self._rbf_sigma = 1.0  # RBF width
        self._rbf_update_interval = 20  # update RBF centers every N steps
        self._rbf_blend = 0.3  # blend factor between linear and RBF predictions
        self._rbf_enabled = True
        self._rbf_warmup = 50  # steps before RBF starts influencing

    def _rbf_predict(self, arm, ctx):
        """Compute RBF-based prediction for given context."""
        if not self._rbf_enabled or self.t < self._rbf_warmup:
            return 0.0
        
        centers = self._rbf_centers[arm]
        weights = self._rbf_weights[arm]
        # Compute RBF activations
        diffs = ctx - centers  # (n_rbf, dim)
        dists_sq = np.sum(diffs ** 2, axis=1)  # (n_rbf,)
        activations = np.exp(-dists_sq / (2.0 * self._rbf_sigma ** 2))
        return float(weights @ activations)

    def _update_rbf(self, arm, ctx, residual):
        """Update RBF centers and weights based on residual."""
        if not self._rbf_enabled or self.t < self._rbf_warmup:
            return
        
        if self.t % self._rbf_update_interval != 0:
            return
            
        centers = self._rbf_centers[arm]
        weights = self._rbf_weights[arm]
        
        # Update weights using gradient descent on residual
        diffs = ctx - centers
        dists_sq = np.sum(diffs ** 2, axis=1)
        activations = np.exp(-dists_sq / (2.0 * self._rbf_sigma ** 2))
        
        # Simple gradient update for RBF weights
        lr = 0.1 / (1.0 + self.t / 100.0)
        weights += lr * residual * activations
        
        # Occasionally update centers to capture new regions
        if self.t % (self._rbf_update_interval * 5) == 0:
            # Move center with largest activation toward current context
            idx = np.argmax(activations)
            centers[idx] = 0.9 * centers[idx] + 0.1 * ctx

    def _adaptive_Q(self, arm, ctx_norm_sq):
        """Compute adaptive process noise based on EMA of squared innovations and context norm."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        scaled = np.tanh(ema_val / 0.4)
        q_base = self._base_Q + (self._Q_max - self._base_Q) * scaled
        # Context-scaled component: larger contexts imply higher potential drift
        q_ctx = self._lambda_ctx * ctx_norm_sq
        return q_base + q_ctx

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft reset: inflate covariance to encourage re-exploration
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm using UCB with Kalman filter estimates."""
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ ctx
            # Add RBF component if enabled
            if self._rbf_enabled and self.t >= self._rbf_warmup:
                mu += self._rbf_blend * self._rbf_predict(a, ctx)
            
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
            
            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update Kalman filter for the selected arm with observed reward."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Prediction step (random walk model) with context-scaled process noise
        ctx_norm_sq = float(x @ x)
        # Update EMA of context norm for this arm
        self._ctx_norm_sq[a] = (1.0 - self._ctx_norm_alpha) * self._ctx_norm_sq[a] + self._ctx_norm_alpha * ctx_norm_sq
        Q = self._adaptive_Q(a, self._ctx_norm_sq[a])
        P_pred = self.P[a] + Q * np.eye(self.d)

        # Update step
        y = r - self.theta[a] @ x

        P_pred_x = P_pred @ x
        S = x @ P_pred_x + self._R
        if S < 1e-12:
            S = 1e-12

        K = P_pred_x / S

        self.theta[a] = self.theta[a] + K * y

        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        # Update RBF model with residual
        if self._rbf_enabled:
            # Compute residual after Kalman update
            pred_after = self.theta[a] @ x
            residual = r - pred_after
            self._update_rbf(a, x, residual)

        self.count[a] += 1
