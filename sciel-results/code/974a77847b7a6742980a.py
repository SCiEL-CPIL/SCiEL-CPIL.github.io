"""Bayesian LinUCB with Cholesky-based solving and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
      - Per-arm ridge regression with exponential forgetting (adaptive decay).
      - UCB exploration with variance-driven alpha.
      - Cholesky-based solves for speed and numerical stability.
      - Time-deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.n_arms = n_arms

        # Regularization
        self.lam = 1.0

        # Per-arm state
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Per-arm running stats for reward variance
        self.count = [0] * n_arms
        self.mean_r = [0.0] * n_arms
        self.m2 = [0.0] * n_arms

        # Base parameters
        self.beta_min = 0.90
        self.beta_max = 0.999
        self.alpha_base = 1.0
        self.var_scale = 2.0

        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 112.0

    def _safe(self):
        return time.time() >= self.deadline

    def _adaptive_beta(self):
        """Compute adaptive decay based on pooled reward variance."""
        total_n = sum(self.count)
        if total_n < 2:
            return 0.99
        pooled_mean = sum(self.mean_r[a] * self.count[a] for a in range(self.n)) / total_n
        pooled_m2 = sum(self.m2[a] + self.count[a] * (self.mean_r[a] - pooled_mean) ** 2
                        for a in range(self.n))
        var = max(pooled_m2 / max(total_n - 1, 1), 1e-8)
        t = min(var / (self.var_scale ** 2), 1.0)
        beta = self.beta_max - t * (self.beta_max - self.beta_min)
        return beta

    def _adaptive_alpha(self):
        """Compute adaptive exploration coefficient."""
        beta = self._adaptive_beta()
        nonstat = 1.0 - beta
        avg_n = max(sum(self.count) / self.n, 1.0)
        conf = 1.0 / np.sqrt(avg_n)
        alpha = self.alpha_base * (0.5 + conf) + 2.0 * nonstat
        return alpha

    def _get_theta_and_cov(self, arm):
        """Return theta and covariance (via Cholesky) for a given arm."""
        try:
            L = np.linalg.cholesky(self.A[arm])
            # Solve A theta = b via Cholesky
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            # Covariance is inv(A), computed via Cholesky
            V = np.linalg.solve(L, np.linalg.inv(np.linalg.solve(L.T, np.eye(self.d))))
            return theta, V
        except np.linalg.LinAlgError:
            # Fallback to direct solve
            try:
                A_inv = np.linalg.inv(self.A[arm])
                theta = A_inv @ self.b[arm]
                return theta, A_inv
            except np.linalg.LinAlgError:
                return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        x = np.asarray(context, dtype=np.float64).flatten()

        if self._safe():
            # Deadline guard: greedy based on current estimates
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta, _ = self._get_theta_and_cov(a)
                    score = theta @ x
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except Exception:
                    pass
            return int(best_arm)

        alpha = self._adaptive_alpha()
        scores = np.zeros(self.n)
        for a in range(self.n):
            theta, V = self._get_theta_and_cov(a)
            mean = theta @ x
            variance = float(x @ V @ x)
            scores[a] = mean + alpha * np.sqrt(max(variance, 0.0))

        # Small random tie-breaker
        scores += np.random.normal(0, 1e-6, size=self.n)
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).flatten()
        r = float(reward)

        if self._safe():
            # Minimal update to keep state valid
            self.A[arm] = self.A[arm] + np.outer(x, x)
            self.b[arm] = self.b[arm] + r * x
            self._update_stats(arm, r)
            return

        beta = self._adaptive_beta()
        # Forgetting: decay old info, add new
        self.A[arm] = beta * self.A[arm] + np.outer(x, x)
        self.b[arm] = beta * self.b[arm] + r * x
        self._update_stats(arm, r)

    def _update_stats(self, arm, reward):
        n = self.count[arm] + 1
        self.count[arm] = n
        delta = reward - self.mean_r[arm]
        self.mean_r[arm] += delta / n
        self.m2[arm] += delta * (reward - self.mean_r[arm])
# EVOLVE-BLOCK-END
