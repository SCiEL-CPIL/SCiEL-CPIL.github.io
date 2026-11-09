# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit with:
    - Per-arm Bayesian linear models with recursive exponential forgetting
    - Thompson Sampling for exploration (posterior sampling)
    - Adaptive forgetting rate based on prediction error
    - EWMA fallback for robustness
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

        # Per-arm Bayesian linear models
        self.gamma = 0.96
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.w_hat = [np.zeros(dim) for _ in range(n_arms)]

        # Thompson Sampling prior strength
        self.alpha_prior = 1.0

        # EWMA per arm for fallback
        self.ewma_alpha = 0.1
        self.ewma_values = np.zeros(n_arms)
        self.ewma_counts = np.zeros(n_arms)

        # Adaptive forgetting tracking
        self.prediction_errors = []
        self.error_window = 20
        self.base_gamma = 0.96
        self.min_gamma = 0.90
        self.max_gamma = 0.99

        # Best arm tracking
        self.arm_recent = {a: [] for a in range(n_arms)}
        self.arm_history_len = 30
        self.best_arm = 0

    def _within_deadline(self):
        return time.time() < self.deadline

    def _sample_weights(self, arm):
        """Sample weights from posterior N(w_hat, (1/alpha) * A_inv) for given arm."""
        cov = (1.0 / self.alpha_prior) * self.A_inv[arm]
        try:
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(self.d))
            z = np.random.randn(self.d)
            return self.w_hat[arm] + L @ z
        except np.linalg.LinAlgError:
            var_diag = np.maximum(np.diag(cov), 1e-8)
            return self.w_hat[arm] + np.sqrt(var_diag) * np.random.randn(self.d)

    def select(self, context):
        self.t += 1

        # Deadline guard
        if not self._within_deadline():
            return self.best_arm

        # First few rounds: pure exploration
        if self.t <= 3:
            return int(np.random.randint(self.n))

        # Thompson Sampling: sample from each arm's posterior and pick max
        scores = np.zeros(self.n)
        for a in range(self.n):
            w_sample = self._sample_weights(a)
            scores[a] = np.dot(w_sample, context)

        # Blend with EWMA for stability after warmup
        if self.t > 15:
            ewma_confidence = min(self.t / 80.0, 0.35)
            score_range = np.max(scores) - np.min(scores)
            if score_range > 1e-8:
                ewma_range = np.max(self.ewma_values) - np.min(self.ewma_values)
                if ewma_range > 1e-8:
                    normalized_ewma = (self.ewma_values - np.min(self.ewma_values)) / ewma_range
                    normalized_scores = (scores - np.min(scores)) / score_range
                    scores = (1.0 - ewma_confidence) * (np.min(scores) + normalized_scores * score_range) + \
                              ewma_confidence * (np.min(self.ewma_values) + normalized_ewma * ewma_range)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if not self._within_deadline():
            return

        # Update per-arm Bayesian linear model with exponential forgetting
        outer = np.outer(context, context)
        self.A[arm] = self.gamma * self.A[arm] + outer
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Update inverse using Sherman-Morrison-Woodbury
        V = self.A_inv[arm]
        Vx = V @ context
        denom = 1.0 / self.gamma + context @ Vx
        if abs(denom) > 1e-12:
            self.A_inv[arm] = (1.0 / self.gamma) * V - (1.0 / (self.gamma * self.gamma)) * np.outer(Vx, Vx) / denom
        else:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d)

        # Update w_hat = A^{-1} b
        self.w_hat[arm] = self.A_inv[arm] @ self.b[arm]

        # Prediction error for adaptive forgetting
        pred = np.dot(self.w_hat[arm], context)
        error = abs(reward - pred)
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.error_window:
            self.prediction_errors.pop(0)

        # Adaptive forgetting: adjust gamma based on recent prediction errors
        if len(self.prediction_errors) >= 5:
            mean_err = np.mean(self.prediction_errors)
            err_scale = min(mean_err / 2.0, 1.0)
            target_gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * err_scale
            target_gamma = max(self.min_gamma, min(self.max_gamma, target_gamma))
            self.gamma = 0.9 * self.gamma + 0.1 * target_gamma

        # Update EWMA
        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1

        # Track arm reward history for best_arm
        self.arm_recent[arm].append(reward)
        if len(self.arm_recent[arm]) > self.arm_history_len:
            self.arm_recent[arm].pop(0)

        # Update best arm based on recent history
        arm_avgs = {}
        for a in range(self.n):
            if self.arm_recent[a]:
                arm_avgs[a] = np.mean(self.arm_recent[a])
            else:
                arm_avgs[a] = -np.inf
        if arm_avgs:
            self.best_arm = max(arm_avgs, key=arm_avgs.get)
# EVOLVE-BLOCK-END
