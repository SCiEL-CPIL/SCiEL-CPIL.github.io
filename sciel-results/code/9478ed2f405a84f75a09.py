"""Non-stationary linear contextual bandit policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with forgetting factor and adaptive reset on large prediction errors."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        # Forgetting factor
        self.gamma = 0.995
        # Exploration parameter
        self.alpha = 1.0
        # Error tracking for adaptive reset
        self.recent_errors = [[] for _ in range(n_arms)]
        self.error_window = 10
        self.error_threshold = 2.0
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin before 120s

    def _check_deadline(self):
        """Return True if we should stop exploring and just exploit."""
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def _update_theta(self, arm):
        """Update theta and inv_A for the given arm."""
        self.inv_A[arm] = np.linalg.inv(self.A[arm])
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def select(self, context):
        # Deadline guard: if near deadline, just exploit best estimate
        if self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv_A = self.inv_A[a]
            mu = theta @ context
            # UCB bonus
            bonus = self.alpha * np.sqrt(max(0.0, context @ inv_A @ context))
            scores.append(mu + bonus)

        # Use soft-max style selection with small epsilon-greedy for exploration
        # Actually, let's use argmax but with a small random tie-breaker
        best_arm = int(np.argmax(scores))
        # Small epsilon-greedy
        if np.random.random() < 0.05:
            best_arm = int(np.random.randint(self.n))
        return best_arm

    def update(self, context, arm, reward):
        # Check deadline
        if self._check_deadline():
            # Just update the model, no exploration
            self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
            self.b[arm] = self.gamma * self.b[arm] + reward * context
            self._update_theta(arm)
            return

        # Prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        abs_error = abs(error)

        # Track recent errors for this arm
        self.recent_errors[arm].append(abs_error)
        if len(self.recent_errors[arm]) > self.error_window:
            self.recent_errors[arm].pop(0)

        # Adaptive reset: if consecutive large errors, inflate covariance
        if len(self.recent_errors[arm]) >= 3:
            recent_avg = np.mean(self.recent_errors[arm][-3:])
            if recent_avg > self.error_threshold:
                # Inflate covariance to reset
                self.A[arm] = self.gamma * self.A[arm] + 0.1 * np.eye(self.d)
                self.b[arm] = self.gamma * self.b[arm]
                self._update_theta(arm)
                # Clear error history after reset
                self.recent_errors[arm] = []

        # Standard update with forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        self._update_theta(arm)
# EVOLVE-BLOCK-END
