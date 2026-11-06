"""Non-stationary linear contextual bandit with adaptive RLS and time-aware exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Per-arm Recursive Least Squares (RLS) with adaptive forgetting
    2. Adaptive forgetting factor based on recent prediction errors
    3. Time-aware exploration bonus for arms not recently played
    4. Deadline guard to avoid timeout
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 145.0  # 145s safety margin before 150s kill

        # Per-arm RLS state: P = inverse of A (covariance-like)
        # Start with high uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Track last time each arm was played
        self.last_played = [-1000] * n_arms

        # Track recent prediction errors for adaptive forgetting
        self.recent_errors = [[] for _ in range(n_arms)]
        self.max_error_history = 50

        # Base forgetting factor
        self.base_forgetting = 0.98

        # Exploration parameter
        self.alpha = 1.0

    def _compute_forgetting(self, arm):
        """Compute adaptive forgetting factor based on recent prediction errors."""
        if len(self.recent_errors[arm]) < 5:
            return self.base_forgetting

        # Use median absolute deviation for robustness
        errors = np.array(self.recent_errors[arm][-self.max_error_history:])
        med = np.median(errors)
        mad = np.median(np.abs(errors - med)) + 1e-8

        # If errors are large relative to median, increase forgetting (faster adaptation)
        # If errors are small, decrease forgetting (more stability)
        error_ratio = np.mean(errors) / (mad + 1e-8)

        # Clamp the ratio
        error_ratio = np.clip(error_ratio, 0.5, 3.0)

        # Higher error ratio -> lower forgetting factor (more forgetting)
        # Map error_ratio from [0.5, 3.0] to forgetting from [0.99, 0.95]
        forgetting = 0.99 - 0.04 * (error_ratio - 0.5) / 2.5
        return np.clip(forgetting, 0.90, 0.995)

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() > self.deadline

    def select(self, context):
        """Select arm using UCB with time-aware exploration bonus."""
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Check deadline - if close, just pick the arm with highest mean estimate
        if self._check_deadline():
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Mean estimate
            mean = self.theta[a] @ context

            # Uncertainty estimate from P
            var = context @ self.P[a] @ context
            uncertainty = np.sqrt(max(var, 0.0))

            # Time-aware exploration bonus: increase if arm hasn't been played recently
            time_since_played = self.t - self.last_played[a]
            time_bonus = 0.3 * np.log1p(time_since_played)

            # UCB score
            score = mean + self.alpha * uncertainty + time_bonus
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update RLS estimate for the given arm."""
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Check deadline
        if self._check_deadline():
            # Minimal update to save time
            self.last_played[arm] = self.t
            return

        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred

        # Store recent errors for adaptive forgetting
        self.recent_errors[arm].append(error)
        if len(self.recent_errors[arm]) > self.max_error_history:
            self.recent_errors[arm].pop(0)

        # Compute adaptive forgetting factor
        gamma = self._compute_forgetting(arm)

        # RLS update with forgetting
        # P = (1/gamma) * P - (1/gamma) * P @ x @ (x^T @ P @ x + gamma)^-1 @ x^T @ P
        Px = self.P[arm] @ context
        denom = context @ Px + gamma
        K = Px / denom  # Gain vector

        # Update theta
        self.theta[arm] = self.theta[arm] + K * error

        # Update P (Joseph form for numerical stability)
        P_old = self.P[arm]
        I = np.eye(self.d)
        P_new = (1.0 / gamma) * P_old - (1.0 / gamma) * np.outer(K, context @ P_old)
        # Symmetrize to reduce numerical drift
        self.P[arm] = (P_new + P_new.T) / 2.0

        # Ensure P remains positive definite-ish by adding small regularization
        self.P[arm] += 1e-6 * np.eye(self.d)

        # Update last played
        self.last_played[arm] = self.t
# EVOLVE-BLOCK-END
