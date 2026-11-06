"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with non-stationary handling, SMW inverse updates, deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # safety margin before 150s kill
        self._recent_errors = []
        self._base_lambda = 0.97
        self._min_lambda = 0.90
        self._max_lambda = 0.995
        self._alpha = 0.5
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda
        recent_mean_error = np.mean(self._recent_errors[-20:])
        error_norm = np.clip(recent_mean_error / 2.0, 0, 1)
        lam = self._max_lambda - error_norm * (self._max_lambda - self._min_lambda)
        return float(lam)

    def _check_deadline(self):
        return time.time() >= self._deadline

    def select(self, context):
        context = np.asarray(context, dtype=float)
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.inv_A[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)
        scores = np.empty(self.n)
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            pred = theta @ context
            var = context @ (self.inv_A[a] @ context)
            ucb = self._alpha * np.sqrt(max(0.0, var))
            scores[a] = pred + ucb
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        # Track prediction error for adaptive forgetting
        theta = self.inv_A[arm] @ self.b[arm]
        pred = theta @ context
        error = abs(reward - pred)
        self._recent_errors.append(error)
        if len(self._recent_errors) > 50:
            self._recent_errors.pop(0)

        lam = self._get_lambda()
        self._round += 1

        # Sherman-Morrison-Woodbury rank-1 update for inv_A
        M = self.inv_A[arm]
        c = context @ (M @ context)
        Mx = M @ context
        self.inv_A[arm] = M / lam - np.outer(Mx, Mx) / (lam * (lam + c))

        # Keep A and b consistent
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        # Periodic recompute to correct SMW drift (cheap, O(d^3) every 500 rounds)
        if self._round % 500 == 0:
            try:
                self.inv_A[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                pass
# EVOLVE-BLOCK-END
