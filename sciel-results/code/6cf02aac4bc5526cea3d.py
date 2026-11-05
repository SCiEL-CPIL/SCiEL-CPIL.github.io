"""Non-stationary linear contextual bandit with unified arm-embedded model and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Unified model: shared context features + arm-specific offsets
    - Adaptive forgetting with smoothed error tracking
    - UCB exploration with error-driven scaling
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_extended = dim + n_arms  # context + one-hot arm indicators
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Unified model state
        self.V = np.eye(self.d_extended) * 1.0  # inverse covariance
        self.b = np.zeros(self.d_extended)       # weighted sum of rewards*x_ext
        self.theta = np.zeros(self.d_extended)   # cached theta = V @ b
        self.count = 0

        # Adaptive parameters
        self.gamma_base = 0.95
        self.gamma_min = 0.80
        self.gamma_max = 0.995
        self.lambda_ridge = 0.1

        # Exploration parameters
        self.C_base = 2.0
        self.C_min = 0.5
        self.C_max = 5.0
        self.round_num = 0

        # Global error tracking
        self.err_ewma = 0.0
        self.err_ewma_alpha = 0.1
        self.err_baseline = None
        self.err_baseline_window = 50
        self.err_history = []

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_C(self, normalized_error):
        """
        Exploration coefficient.
        Scales with the estimated error/drift level.
        """
        t = self.round_num
        
        # Base decay over time
        decay = 1.0 / (1.0 + t / 200.0)
        C_time = self.C_min + (self.C_base - self.C_min) * decay
        
        # Error-driven scaling
        error_scale = 1.0 + 1.5 * min(normalized_error, 2.0)
        
        C_val = C_time * error_scale
        
        # Ensure strong exploration early on
        if self.count < 3 * self.n:
            C_val = max(C_val, self.C_max * 0.8)
            
        return min(C_val, self.C_max)

    def _adaptive_gamma(self, current_err_sq):
        """
        Adapt forgetting factor based on smoothed error signal.
        """
        alpha = self.err_ewma_alpha
        if self.err_ewma == 0.0 and self.count < 5:
            self.err_ewma = current_err_sq
        else:
            self.err_ewma = (1 - alpha) * self.err_ewma + alpha * current_err_sq
        
        # Maintain a history of recent errors for baseline calculation
        self.err_history.append(current_err_sq)
        if len(self.err_history) > self.err_baseline_window:
            self.err_history.pop(0)
            
        # Calculate baseline
        if len(self.err_history) >= 10:
            baseline = np.mean(self.err_history)
            if self.err_baseline is None:
                self.err_baseline = baseline
            # Smoothly update baseline
            self.err_baseline = 0.95 * self.err_baseline + 0.05 * baseline
        else:
            baseline = self.err_ewma if self.err_ewma > 0 else 1e-6

        # Determine gamma
        gamma = self.gamma_base
        
        if baseline > 1e-8:
            ratio = self.err_ewma / baseline
            
            if ratio > 1.2:
                reduction = 0.15 * min(ratio - 1.0, 3.0)
                gamma -= reduction
            elif ratio < 0.8:
                increase = 0.05 * (1.0 - ratio)
                gamma += increase

        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _extend_context(self, context, arm):
        """
        Create extended feature vector: [context, one-hot arm indicator]
        """
        x_ext = np.zeros(self.d_extended)
        x_ext[:self.d] = context
        x_ext[self.d + arm] = 1.0
        return x_ext

    def _update_theta(self):
        """Update cached theta."""
        self.theta = self.V @ self.b

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                x_ext = self._extend_context(context, a)
                scores.append(self.theta @ x_ext)
            return int(np.argmax(scores))

        # Calculate normalized error for exploration scaling
        if self.err_baseline is not None and self.err_baseline > 1e-8:
            norm_err = self.err_ewma / self.err_baseline
        else:
            norm_err = 1.0
        
        c_val = self._get_C(norm_err)
        
        scores = []
        for a in range(self.n):
            x_ext = self._extend_context(context, a)
            mu = self.theta @ x_ext
            
            # UCB bonus: c * sqrt(x^T V x)
            Vx = self.V @ x_ext
            var_term = x_ext @ Vx
            if var_term < 0:
                var_term = 0.0
            bonus = c_val * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1

        x_ext = self._extend_context(context, arm)
        
        # Track prediction error for adaptive gamma
        pred = self.theta @ x_ext
        pred_err = reward - pred
        pred_err_sq = pred_err * pred_err
        
        # Adaptive forgetting
        gamma = self._adaptive_gamma(pred_err_sq)

        # Update inverse covariance with forgetting + ridge
        # Using Sherman-Morrison:
        # V_new = (1/gamma) * V_old - (1/gamma) * (V_old x x^T V_old) / (gamma + x^T V_old x)
        
        V_old = self.V
        Vx = V_old @ x_ext
        xVx = x_ext @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-10:
            V_new = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback to regularized identity if singular
            V_new = np.eye(self.d_extended) * (1.0 / self.lambda_ridge)

        # Symmetrize for numerical stability
        V_new = 0.5 * (V_new + V_new.T)

        # Clip: ensure positive semi-definite by adding small ridge if needed
        diag = np.diag(V_new)
        if np.any(diag < 1e-8) or np.any(~np.isfinite(diag)):
            V_new = np.eye(self.d_extended) * (1.0 / self.lambda_ridge)
            V_new = 0.5 * (V_new + V_new.T)

        self.V = V_new

        # Update b with forgetting
        self.b = gamma * self.b + reward * x_ext

        # Update theta
        self._update_theta()
        
        # Increment count
        self.count += 1
# EVOLVE-BLOCK-END
