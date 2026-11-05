"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Structurally different approach: Per-arm exponential recency-weighted
    ridge regression with adaptive window, plus a lightweight exploration
    bonus based on prediction uncertainty. Uses a sliding window of recent
    observations per arm rather than discounted sufficient statistics,
    and re-estimates parameters only when needed.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim

        # Per-arm history of (context, reward) pairs
        self.histories = [[] for _ in range(n_arms)]
        self.max_history = 50  # sliding window size

        # Adaptive window: start wide, shrink when errors are high
        self.window_size = 50
        self.min_window = 10
        self.max_window = 50

        # Per-arm cached parameters
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        self.covs = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # covariance (inverse precision)
        self.needs_update = [True] * n_arms

        # Exploration bonus coefficient
        self.alpha = 1.0

        # Error tracking for adaptive window
        self.recent_errors = []
        self.error_window = 15

        # Timing
        self._start_time = time.time()
        self._deadline = self._start_time + 110

        # Precompute identity
        self.eye = np.eye(dim)

    def _estimate_arm(self, a):
        """Estimate parameters for arm a from its history using weighted ridge regression."""
        h = self.histories[a]
        if len(h) == 0:
            self.thetas[a] = np.zeros(self.dim)
            self.covs[a] = self.eye.copy()
            return

        # Use most recent window_size observations
        n = min(len(h), self.window_size)
        X = np.array([h[i][0] for i in range(len(h) - n, len(h))])
        y = np.array([h[i][1] for i in range(len(h) - n, len(h))])

        # Weighted ridge: older observations get lower weight
        # Weights: exponential decay from 1.0 (most recent) to small value (oldest)
        if n > 1:
            weights = np.exp(np.linspace(0, -2.0, n))  # decay factor
            weights = weights / weights.sum()
        else:
            weights = np.array([1.0])

        # Weighted least squares with ridge regularization
        W = np.diag(weights)
        # theta = (X^T W X + lambda*I)^-1 X^T W y
        lam = 1.0 / max(n, 1)  # regularization strength
        XtWX = X.T @ W @ X + lam * self.eye
        XtWy = X.T @ W @ y

        try:
            self.thetas[a] = np.linalg.solve(XtWX, XtWy)
            # Covariance approximation: (X^T W X + lambda*I)^-1
            self.covs[a] = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            self.thetas[a] = np.zeros(self.dim)
            self.covs[a] = self.eye.copy()

        self.needs_update[a] = False

    def _adapt_window(self, arm, context, reward):
        """Adapt window size based on prediction error."""
        # Predict using current estimate
        pred = self.thetas[arm] @ context
        error = abs(reward - pred)
        self.recent_errors.append(error)

        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        if len(self.recent_errors) >= 5:
            avg_err = np.mean(self.recent_errors)
            if avg_err > 1.5:
                # High error: shrink window to focus on recent data
                self.window_size = max(self.min_window, int(self.window_size * 0.85))
            elif avg_err < 0.5:
                # Low error: can expand window
                self.window_size = min(self.max_window, int(self.window_size * 1.05))

    def select(self, context):
        """Select arm using mean reward + exploration bonus."""
        if time.time() > self._deadline:
            # Emergency: pick best by current mean estimate
            scores = [self.thetas[a] @ context for a in range(self.n_arms)]
            return int(np.argmax(scores))

        # Update estimates if needed
        for a in range(self.n_arms):
            if self.needs_update[a] and self.histories[a]:
                self._estimate_arm(a)

        scores = []
        for a in range(self.n_arms):
            mean_reward = self.thetas[a] @ context
            # Exploration bonus: sqrt(context^T Cov context)
            try:
                var = context @ self.covs[a] @ context
                bonus = self.alpha * np.sqrt(max(var, 0))
            except Exception:
                bonus = 0.0
            scores.append(mean_reward + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model with observed reward."""
        self._adapt_window(arm, context, reward)

        # Add to history
        self.histories[arm].append((context, reward))

        # Trim history if too long
        if len(self.histories[arm]) > self.max_history * 2:
            self.histories[arm] = self.histories[arm][-self.max_history:]

        # Mark for update
        self.needs_update[arm] = True

        # Re-estimate immediately for this arm (lightweight for small dim)
        if len(self.histories[arm]) <= self.window_size:
            self._estimate_arm(arm)
# EVOLVE-BLOCK-END
