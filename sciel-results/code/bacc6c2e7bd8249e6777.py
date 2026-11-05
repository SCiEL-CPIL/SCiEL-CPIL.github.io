"""Non-stationary linear contextual bandit with shared model and arm-specific basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Shared linear model with arm-specific basis functions (one-hot + shared features)
    - Exponential forgetting (discounted least squares) for non-stationarity
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with unified uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Extended dimension: dim (shared) + n_arms (one-hot)
        self.d_ext = dim + n_arms

        # Shared model: V = (sum gamma^t x_ext x_ext^T + lambda I)^{-1}, b = sum gamma^t r x_ext
        self.V = np.eye(self.d_ext) * 1.0  # Start with identity (inverse of I)
        self.b = np.zeros(self.d_ext)
        self.counts = [0] * n_arms
        self.total_count = 0

        # For adaptive forgetting
        self.prediction_errors = []
        self.max_err_window = 50

        # Exploration
        self.C = 2.0
        self.base_gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.99

        # Ridge parameter for stability
        self.ridge = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _extend_context(self, context, arm):
        """Create extended context with one-hot arm indicator."""
        x_ext = np.zeros(self.d_ext)
        x_ext[:self.d] = context
        x_ext[self.d + arm] = 1.0
        return x_ext

    def _adaptive_gamma(self):
        """Adapt forgetting factor based on recent prediction errors."""
        c = self.total_count
        if c < 10:
            return self.base_gamma

        # Use recent prediction errors to detect non-stationarity
        if len(self.prediction_errors) < 10:
            return self.base_gamma

        # Compare recent errors to earlier ones
        n = len(self.prediction_errors)
        recent_mean = np.mean(self.prediction_errors[-15:])
        older_mean = np.mean(self.prediction_errors[:max(1, n - 15)])

        if older_mean < 1e-12:
            ratio = 1.0
        else:
            ratio = recent_mean / older_mean

        # If errors are increasing, reduce gamma (faster forgetting)
        # If errors are stable or decreasing, increase gamma (slower forgetting)
        gamma = self.base_gamma
        if ratio > 1.2:
            gamma = self.gamma_min + (self.base_gamma - self.gamma_min) * (1.0 / min(ratio, 5.0))
        elif ratio < 0.8:
            gamma = min(self.gamma_max, self.base_gamma + 0.03 * (1.0 - ratio))

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self):
        """Adapt exploration coefficient based on total count."""
        c = self.total_count
        if c < 10:
            return self.C * 2.5
        if c < 50:
            return self.C * 1.5
        return self.C

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            theta = self.V @ self.b
            scores = []
            for a in range(self.n):
                x_ext = self._extend_context(context, a)
                scores.append(theta @ x_ext)
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        theta = self.V @ self.b
        scores = []
        c_val = self._adaptive_C()
        for a in range(self.n):
            x_ext = self._extend_context(context, a)
            mu = theta @ x_ext
            bonus = c_val * np.sqrt(x_ext @ self.V @ x_ext)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.total_count += 1

        x_ext = self._extend_context(context, arm)

        # Compute prediction error before updating
        theta = self.V @ self.b
        pred = theta @ x_ext
        err = abs(reward - pred)
        self.prediction_errors.append(err)
        if len(self.prediction_errors) > self.max_err_window:
            self.prediction_errors.pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma()

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V
        Vx = V_old @ x_ext
        xVx = x_ext @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V = np.eye(self.d_ext)

        # Update b with forgetting
        self.b = gamma * self.b + reward * x_ext
# EVOLVE-BLOCK-END
