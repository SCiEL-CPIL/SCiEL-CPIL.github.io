"""Non-stationary linear contextual bandit with per-arm ridge regression,
caching, adaptive forgetting, and adaptive UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear bandit with per-arm ridge regression.
    - Caches theta and A^{-1} for speed
    - Adaptive forgetting factor based on recent prediction errors
    - Adaptive exploration coefficient (alpha) scaled with uncertainty
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._start_time = time.time()
        self._deadline = 118.0  # safety margin before 120s limit

        # Per-arm linear models
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # Regularized covariance
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Cached inverses and theta
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive parameters
        self.base_forgetting = 0.95
        self.base_alpha = 1.0

        # Track recent prediction errors for adaptivity
        self.recent_errors = []
        self.error_window = 15

        # Round counter
        self.round = 0

    def _recompute_cache(self, arm):
        """Recompute A_inv and theta for a given arm."""
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            try:
                self.A_inv[arm] = np.linalg.pinv(self.A[arm])
                self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            except Exception:
                self.A_inv[arm] = np.eye(self.d)
                self.theta[arm] = np.zeros(self.d)

    def _get_adaptive_params(self):
        """Compute adaptive forgetting factor and exploration coefficient."""
        if len(self.recent_errors) < 3:
            return self.base_forgetting, self.base_alpha

        recent = self.recent_errors[-self.error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent)

        # Adaptive forgetting: more forgetting when errors are high
        # Map mean error to forgetting factor in [0.85, 0.99]
        if mean_err > 1.0:
            excess = min(mean_err / 1.0, 3.0)
            forgetting = self.base_forgetting - 0.1 * min(excess, 1.0)
            forgetting = max(0.85, min(0.99, forgetting))
        elif mean_err < 0.1:
            ratio = mean_err / 0.1
            forgetting = min(0.995, self.base_forgetting + 0.03 * (1.0 - ratio))
        else:
            t = (mean_err - 0.1) / (1.0 - 0.1)
            forgetting = self.base_forgetting - 0.04 * t

        # Adaptive alpha: increase exploration when errors are high
        if mean_err > 1.0:
            excess = min(mean_err / 1.0, 2.0)
            alpha = self.base_alpha + 0.8 * min(excess, 1.5)
        elif mean_err < 0.1:
            ratio = mean_err / 0.1
            alpha = max(0.4, self.base_alpha - 0.4 * (1.0 - ratio))
        else:
            t = (mean_err - 0.1) / (1.0 - 0.1)
            alpha = self.base_alpha + 0.3 * t

        # Small volatility component
        alpha += 0.15 * min(std_err / max(mean_err, 1e-8), 1.0)

        return forgetting, alpha

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: return greedy best
        if time.time() - self._start_time > self._deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                score = float(self.theta[a] @ context)
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        forgetting, alpha = self._get_adaptive_params()

        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = float(self.theta[a] @ context)
            var = float(context @ self.A_inv[a] @ context)
            bonus = alpha * np.sqrt(max(var, 0.0))
            scores[a] = mean + bonus

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.round += 1

        # Compute prediction error before update
        pred = float(self.theta[arm] @ context)
        error = reward - pred

        # Track recent errors
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.error_window * 2:
            self.recent_errors = self.recent_errors[-self.error_window * 2:]

        # Adaptive forgetting
        forgetting, _ = self._get_adaptive_params()

        # Update model
        self.A[arm] = forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = forgetting * self.b[arm] + reward * context

        # Recompute cache for this arm
        self._recompute_cache(arm)

        # Periodic re-regularization for all arms to prevent numerical drift
        if self.round % 50 == 0:
            for a in range(self.n):
                self.A[a] += 1e-4 * np.eye(self.d)
                self._recompute_cache(a)
# EVOLVE-BLOCK-END
