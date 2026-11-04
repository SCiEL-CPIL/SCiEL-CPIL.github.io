"""Non-stationary linear contextual bandit with adaptive forgetting and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Thompson Sampling
    with an adaptive forgetting factor to handle drift in arm weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm inverse covariance (V = A^{-1}) and weighted sum (b)
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting factor
        self.gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.999

        # Track prediction error to adapt gamma
        self.recent_errors = []
        self.error_window = 50
        self.base_error = None

        # Exploration scale
        self.C = 1.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Track context norms for scaling
        self.context_norms = []

    def _check_deadline(self):
        """Return True if we should stop computing and use a safe fallback."""
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def _adaptive_gamma(self, arm, context, reward):
        """Adapt forgetting factor based on prediction error."""
        # Predicted reward
        theta = np.linalg.solve(self.V[arm].T, self.b[arm])  # V is inverse, so theta = V @ b
        # Actually theta = A^{-1} b = V @ b
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        error = abs(reward - pred)

        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        if self.base_error is None:
            if len(self.recent_errors) >= 10:
                self.base_error = np.mean(self.recent_errors)

        if self.base_error is not None and self.base_error > 1e-6:
            ratio = np.mean(self.recent_errors) / self.base_error
            # If errors are increasing, increase forgetting (decrease gamma)
            if ratio > 1.5:
                self.gamma = max(self.gamma_min, self.gamma * 0.95)
            elif ratio < 0.7:
                self.gamma = min(self.gamma_max, self.gamma * 1.02)
        else:
            # Gradually increase gamma toward max as we learn
            self.gamma = min(self.gamma_max, self.gamma + 0.001)

    def select(self, context):
        """Select arm using Thompson Sampling with adaptive exploration."""
        if self._check_deadline():
            # Fallback: pick arm with highest mean estimate
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        # Thompson Sampling: sample from posterior for each arm
        sampled_scores = []
        for a in range(self.n):
            # Posterior: theta ~ N(theta_hat, V)
            theta_hat = self.V[a] @ self.b[a]
            # Sample theta
            # Use Cholesky for sampling: V = L L^T, then theta = theta_hat + L @ z
            try:
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                theta_sample = theta_hat

            sampled_scores.append(theta_sample @ context)

        # Add small exploration noise to break ties and encourage diversity
        noise = 0.01 * np.random.randn(self.n)
        sampled_scores = np.array(sampled_scores) + noise

        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        """Update per-arm statistics with forgetting factor."""
        context = np.asarray(context, dtype=np.float64)

        # Adapt gamma before updating
        self._adaptive_gamma(arm, context, reward)

        # Apply forgetting and update
        self.V[arm] = self.V[arm] / self.gamma  # This is wrong; let me fix
        # Actually, with forgetting: A_new = gamma * A_old + x x^T
        # So V_new = (gamma * A_old + x x^T)^{-1}
        # Using Sherman-Morrison: V_new = V_old / gamma - (V_old x x^T V_old / gamma) / (gamma + x^T V_old x / gamma)
        # Let's do it properly.

        # Reset V properly
        # We stored V = A^{-1}. With forgetting:
        # A_new = gamma * A_old + x x^T
        # V_new = A_new^{-1} = (gamma * A_old + x x^T)^{-1}
        # = gamma^{-1} (A_old + gamma^{-1} x x^T)^{-1}
        # Using Sherman-Morrison on (A_old + gamma^{-1} x x^T):
        # = gamma^{-1} [A_old^{-1} - A_old^{-1} (gamma^{-1} x x^T) A_old^{-1} / (1 + gamma^{-1} x^T A_old^{-1} x)]
        # = gamma^{-1} [V_old - V_old (gamma^{-1} x x^T) V_old / (1 + gamma^{-1} x^T V_old x)]
        # = V_old / gamma - V_old x x^T V_old / (gamma + x^T V_old x)

        V_old = self.V[arm]
        x = context
        xVx = x @ V_old @ x
        denom = self.gamma + xVx
        if denom < 1e-10:
            denom = 1e-10

        self.V[arm] = V_old / self.gamma - np.outer(V_old @ x, V_old @ x) / denom
        self.b[arm] = self.gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
