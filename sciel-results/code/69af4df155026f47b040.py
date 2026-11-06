"""Non-stationary linear contextual bandit with global model + arm biases and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - A single global linear model shared across arms for context features
    - Arm-specific bias terms for per-arm offset adaptation
    - Exponential forgetting with adaptive lambda based on recent prediction errors
    - UCB exploration with time-aware bonus
    - Deadline guard to avoid timeout
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Global linear model (shared across arms)
        self.A_global = np.eye(dim)  # dim x dim
        self.b_global = np.zeros(dim)  # dim

        # Arm-specific biases
        self.bias = np.zeros(n_arms)
        self.bias_counts = np.zeros(n_arms)

        # Per-arm tracking for time-aware exploration
        self.last_played = np.full(n_arms, -1)
        self.arm_counts = np.zeros(n_arms, dtype=np.int64)

        # Adaptive forgetting: track recent errors
        self.recent_errors = []
        self.error_window = 50

        # Initial parameters
        self.base_lambda = 0.98  # base forgetting factor
        self.base_alpha = 1.0  # initial regularization (inverse of P scale)
        self.exploration_base = 0.5
        self.exploration_scale = 0.3

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s

    def _check_deadline(self):
        """Check if we're close to the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def _compute_adaptive_lambda(self):
        """
        Compute adaptive forgetting factor based on recent prediction errors.
        Higher errors -> lower lambda (faster forgetting to adapt to non-stationarity).
        Uses a smooth logistic mapping for stability.
        """
        if len(self.recent_errors) < 5:
            return self.base_lambda

        recent = np.array(self.recent_errors[-self.error_window:])
        mean_abs_error = np.mean(np.abs(recent))

        # Map mean absolute error to lambda in [0.95, 0.999]
        # Low error -> high lambda (slow forgetting), High error -> low lambda (fast forgetting)
        # Use logistic: lambda = 0.999 - 0.049 * sigmoid(k * (error - threshold))
        threshold = 0.5
        k = 3.0
        sigmoid_val = 1.0 / (1.0 + np.exp(-k * (mean_abs_error - threshold)))
        lam = 0.999 - 0.049 * sigmoid_val
        lam = np.clip(lam, 0.95, 0.999)
        return float(lam)

    def _compute_adaptive_alpha(self):
        """
        Compute adaptive exploration strength based on recent performance.
        Higher uncertainty or recent errors -> more exploration.
        """
        if len(self.recent_errors) < 5:
            return self.base_alpha

        recent = np.array(self.recent_errors[-self.error_window:])
        mean_abs_error = np.mean(np.abs(recent))

        # Scale exploration: base + scale * sigmoid(error)
        threshold = 0.5
        k = 2.0
        sigmoid_val = 1.0 / (1.0 + np.exp(-k * (mean_abs_error - threshold)))
        alpha = self.base_alpha + self.exploration_scale * sigmoid_val
        return float(alpha)

    def _get_theta_global(self):
        """Compute global theta = A_global^{-1} b_global."""
        try:
            return np.linalg.solve(self.A_global, self.b_global)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def _get_uncertainty(self, context, theta_global):
        """
        Compute uncertainty for a context given the global model.
        Uses the diagonal of A_global^{-1} projected onto context.
        """
        try:
            A_inv = np.linalg.inv(self.A_global)
            var = context @ A_inv @ context
            return np.sqrt(max(0.0, var))
        except np.linalg.LinAlgError:
            return 1.0

    def select(self, context):
        """Select arm using UCB with global model + arm biases."""
        # Deadline guard: if close to deadline, just pick the best known arm
        if self._check_deadline():
            # Fallback: pick arm with highest estimated reward
            theta = self._get_theta_global()
            scores = np.array([theta @ context + self.bias[a] for a in range(self.n)])
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        theta = self._get_theta_global()
        alpha = self._compute_adaptive_alpha()

        scores = []
        for a in range(self.n):
            # Mean reward estimate
            mean = theta @ context + self.bias[a]

            # Uncertainty from global model
            unc = self._get_uncertainty(context, theta)

            # Time-aware bonus: increase exploration for arms not recently played
            time_since_played = self.t - self.last_played[a]
            if self.last_played[a] < 0:
                time_bonus = self.exploration_base * 2.0
            else:
                # Decay the bonus over time since last play
                time_bonus = self.exploration_base * np.exp(-0.05 * time_since_played)

            # Total score: mean + uncertainty bonus + time bonus
            score = mean + alpha * unc + time_bonus
            scores.append(score)

        scores = np.array(scores)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global model and arm-specific bias."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Compute prediction error for adaptive parameters
        theta = self._get_theta_global()
        predicted = theta @ context + self.bias[arm]
        error = reward - predicted
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        # Adaptive forgetting factor
        lam = self._compute_adaptive_lambda()

        # Update global model with forgetting
        # A = lam * A + x x^T
        # b = lam * b + y * x
        self.A_global = lam * self.A_global + np.outer(context, context)
        self.b_global = lam * self.b_global + reward * context

        # Update arm-specific bias
        # Simple exponential moving average for bias
        self.bias_counts[arm] += 1
        if self.bias_counts[arm] == 1:
            self.bias[arm] = reward - theta @ context
        else:
            # Exponential moving average with same forgetting factor
            self.bias[arm] = lam * self.bias[arm] + (1 - lam) * (reward - theta @ context)

        # Track last played time
        self.last_played[arm] = self.t
        self.arm_counts[arm] += 1
# EVOLVE-BLOCK-END
