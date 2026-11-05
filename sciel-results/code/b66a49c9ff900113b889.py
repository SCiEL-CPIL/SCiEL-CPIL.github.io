# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Per-Arm Recursive Least Squares (RLS)
    with Context-Adaptive Forgetting Factor, Huber Robustification, and UCB Exploration.

    Design Rationale:
    1. Structure: Per-arm RLS filters (Theta, P) for direct, efficient parameter tracking.
       This avoids the overhead of inverting a large unified precision matrix (d*n x d*n)
       and allows for independent, specialized adaptation per arm.
    2. Adaptation: Context-Adaptive Forgetting Factor (lambda).
       Hypothesis: High-magnitude contexts correlate with higher reward volatility or
       faster drift. We scale lambda_t = 1 - (base_forgetting + gamma * ||x||^2).
       Lower lambda (faster forgetting) is applied when context norms are large.
    3. Robustness: Huber-loss innovation clipping to prevent outliers from destabilizing
       the covariance matrix.
    4. Recovery: Sliding-window covariance reset for catastrophic forgetting recovery.
    5. Exploration: Adaptive UCB with warmup and decay.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        # Safety margin: 110s allows 10s buffer before the 120s kill
        self.deadline = self.start_time + 110.0 

        # RLS State per arm
        # Theta: weight vectors (d,)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # P: inverse gain matrices (d x d)
        # Initialize with high uncertainty to encourage exploration
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Context-Adaptive Forgetting Factor Parameters
        # lambda_t = 1 - delta_t
        # delta_t = base_delta + gamma * ||x_t||^2
        # We want lambda in (0, 1). Smaller lambda means faster forgetting.
        # Let's define lambda directly as a function of context norm.
        # lambda_t = 1 / (1 + alpha * ||x_t||^2) ? Or simply 1 - c*||x||^2?
        # The prompt suggests: lambda_t = lambda_base + gamma * ||x_t||^2
        # Wait, usually "forgetting factor" lambda is close to 1.
        # If lambda increases, we forget LESS (remember more).
        # If lambda decreases, we forget MORE (adapt faster).
        # The prompt says: "dynamically scaled by the magnitude... lambda_t = lambda_base + gamma ||x_t||^2"
        # If gamma > 0, high norm -> high lambda -> slower forgetting?
        # "hypothesizing that high-magnitude contexts correlate with higher reward volatility or faster drift, thereby accelerating adaptation"
        # Accelerating adaptation means FORGETTING FASTER.
        # Forgetting faster means lambda should be SMALLER.
        # So if the prompt formula adds gamma * norm^2, and we want acceleration (smaller lambda),
        # then gamma should be negative? Or the formula implies lambda is the "retention" factor?
        # Standard RLS: P_t^{-1} = lambda * P_{t-1}^{-1} + x x^T
        # If lambda is close to 1, we retain old info. If lambda < 1, we forget.
        # To accelerate adaptation (forget faster), we need SMALLER lambda.
        # So we want lambda_t to be SMALL when ||x|| is LARGE.
        # The prompt formula: lambda_t = lambda_base + gamma ||x_t||^2.
        # To satisfy "accelerate adaptation" (smaller lambda) with "high magnitude" (larger norm),
        # gamma must be NEGATIVE.
        # Let's implement: lambda_t = 1 - (delta_base + gamma_pos * ||x||^2)
        # Or simply: lambda_t = 1 / (1 + k * ||x||^2)
        # Let's stick to the prompt's structure but ensure the sign logic is correct for "acceleration".
        # Let's use: lambda_t = 1 - (0.01 + 0.005 * min(||x||^2, 10.0))
        # This ensures lambda < 1 and decreases as norm increases.
        
        self.lambda_base = 0.99       # Base retention (1 - 0.01)
        self.gamma_norm = 0.002       # Scaling factor for norm penalty
        self.norm_cap = 50.0          # Cap on ||x||^2 to prevent lambda from going too low/negative
        
        # Huber Robustification
        self.huber_delta = 2.0        # Threshold in std units

        # Sliding-window Reset
        self.reset_threshold_scale = 2.5
        self.reset_cov_scale = 50.0
        self.reset_window = 8
        self.consecutive_large_innov = [0] * n_arms

        # UCB Exploration
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0

        # Pre-compute identity for speed
        self.I = np.eye(dim)

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _get_lambda(self, ctx):
        """
        Compute adaptive forgetting factor lambda.
        High context norm -> lower lambda -> faster forgetting/adaptation.
        """
        norm_sq = float(np.dot(ctx, ctx))
        norm_sq = min(norm_sq, self.norm_cap)
        # lambda = 1 - (base_forgetting + gamma * norm_sq)
        # base_forgetting = 0.01
        # gamma = 0.002
        # If norm_sq=0, lambda = 0.99
        # If norm_sq=50, lambda = 1 - (0.01 + 0.1) = 0.89
        # This provides a 10% increase in forgetting rate for high-norm contexts.
        delta = 0.01 + self.gamma_norm * norm_sq
        # Ensure lambda stays in (0, 1)
        # Max delta = 0.01 + 0.002*50 = 0.11 -> lambda = 0.89
        # Min delta = 0.01 -> lambda = 0.99
        lambda_t = 1.0 - delta
        return lambda_t

    def _huber_innov(self, innov, std):
        """
        Robustify innovation using Huber loss logic.
        """
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, ctx):
        """
        RLS Prediction/Adaptation Step.
        Update P to account for non-stationarity (forgetting).
        P_t = (1/lambda) * P_{t-1}
        """
        lambda_t = self._get_lambda(ctx)
        # Inverse gain update: P_new = (1/lambda) * P_old
        # This increases uncertainty (P grows) as we forget old data.
        self.P[a] = self.P[a] / lambda_t
        # Add jitter for stability
        self.P[a] += self.jitter * self.I

    def _update(self, a, ctx, reward):
        """
        RLS Update Step.
        1. Calculate innovation.
        2. Robustify innovation.
        3. Calculate Kalman Gain K = P x / (1 + x^T P x).
        4. Update Theta.
        5. Update P (Joseph form or standard rank-1).
        """
        P = self.P[a]
        theta = self.theta[a]
        
        # Predicted reward
        pred = float(theta @ ctx)
        # Raw innovation
        innov_raw = reward - pred
        
        # Innovation variance: S = 1 + x^T P x (assuming R=1)
        var = float(ctx @ P @ ctx) + 1.0 + self.jitter
        std = float(np.sqrt(var))
        
        # Robustify innovation
        innov_robust = self._huber_innov(innov_raw, std)
        
        # Kalman Gain: K = P x / S
        K = (P @ ctx) / var
        
        # Update mean: theta <- theta + K * innov_robust
        self.theta[a] = theta + (K * innov_robust)
        
        # Update P (Inverse Gain)
        # Standard RLS: P_new = P_old - K x^T P_old
        # Joseph Form for stability: P_new = (I - K x^T) P_old (I - K x^T)^T + R K K^T ?
        # Actually, for RLS with R=1, the Joseph form is:
        # P_new = (I - K x^T) P_old
        # Let's use the standard rank-1 update which is very stable for RLS:
        # P_new = P_old - (P_old x x^T P_old) / (1 + x^T P_old x)
        # Which is equivalent to: P_new = P_old - K x^T P_old
        
        IK = self.I - np.outer(K, ctx)
        # Joseph form: P_new = IK @ P_old @ IK.T + R * K @ K.T ?
        # No, the standard Joseph form for covariance P (not inverse) is:
        # P_new = (I - K H) P (I - K H)^T + K R K^T
        # Here H=x, R=1.
        # P_new = (I - K x^T) P (I - K x^T)^T + K K^T
        # However, in RLS, the "observation noise" is absorbed into the normalization.
        # The simpler and very stable update for the inverse covariance P in RLS is:
        # P_new = P_old - K x^T P_old
        
        # Let's use the Joseph form for the covariance P (which is what we store) to ensure PSD.
        # Note: K here is the gain.
        # P_new = (I - K x^T) P_old (I - K x^T)^T + K K^T
        # This is computationally heavier but more stable.
        # Given d is small (likely 5-10), this is fine.
        
        self.P[a] = IK @ P @ IK.T + np.outer(K, K)
        
        # Symmetrize
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        
        # Ensure PSD by clipping small negative eigenvalues (approx via diagonal)
        diag = np.diag(self.P[a])
        if np.any(diag < 0):
            np.fill_diagonal(self.P[a], np.maximum(diag, 1e-6))
            
        self.count[a] += 1
        return innov_raw

    def _check_reset(self, a, innov_raw):
        """
        Check if we should reset covariance due to sustained large innovations.
        """
        P = self.P[a]
        var = float(np.trace(P)) / self.d 
        std = float(np.sqrt(max(var, 0.0)))
        threshold = self.reset_threshold_scale * std + 0.5

        if abs(innov_raw) > threshold:
            self.consecutive_large_innov[a] += 1
        else:
            self.consecutive_large_innov[a] = 0

        if self.consecutive_large_innov[a] >= self.reset_window:
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
            return True
        return False

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if self._check_deadline():
            # Fallback: Greedy
            scores = [float(self.theta[a] @ ctx) for a in range(self.n)]
            return int(np.argmax(scores))

        self.step_count += 1
        
        # Perform prediction step for all arms to update their P matrices
        # This is necessary because RLS requires the prediction step before the update step
        # to account for the passage of time/non-stationarity.
        for a in range(self.n):
            self._predict(a, ctx)

        u = self._adaptive_u()
        scores = []
        
        for a in range(self.n):
            theta_a = self.theta[a]
            P_a = self.P[a]
            
            mean = float(theta_a @ ctx)
            
            # Uncertainty: sqrt(x^T P x)
            var = float(ctx @ P_a @ ctx)
            uncertainty = float(np.sqrt(max(0.0, var)))
            
            # Count-based exploration bonus
            count_factor = 1.0 / (1.0 + self.count[a] * self.count_scale)
            bonus = u * uncertainty * count_factor + 0.1 * count_factor
            
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        
        # Perform the RLS update for the selected arm
        innov_raw = self._update(arm, ctx, reward)
        
        # Check for reset condition
        self._check_reset(arm, innov_raw)
# EVOLVE-BLOCK-END
