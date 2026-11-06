"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with Sherman-Morrison-Woodbury rank-1 updates and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0
        self._base_lambda = 0.95
        self._min_lambda = 0.90
        self._max_lambda = 0.995
        self._recent_errors = []
        self._alpha = 0.5
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda
        recent_mean_error = np.mean(self._recent_errors[-20:])
        # Map error to lambda: higher error -> lower lambda (more forgetting)
        # Normalize error roughly to [0, 1]
        norm_err = min(recent_mean_error / 2.0, 1.0)
        lam = self._max_lambda - norm_err * (self._max_lambda - self._min_lambda)
        return max(self._min_lambda, min(self._max_lambda, lam))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mean = theta @ context
            # UCB bonus
            bonus = self._alpha * np.sqrt(context @ self.A_inv[a] @ context)
            scores.append(mean + bonus)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        # Track prediction error for adaptive lambda
        if arm is not None:
            theta = self.A_inv[arm] @ self.b[arm]
            pred = theta @ context
            err = (reward - pred) ** 2
            self._recent_errors.append(err)
            if len(self._recent_errors) > 100:
                self._recent_errors = self._recent_errors[-100:]

        lam = self._get_lambda()

        # Decay A_inv and b
        self.A_inv[arm] = lam * self.A_inv[arm]
        self.b[arm] = lam * self.b[arm]

        # Rank-1 update using Sherman-Morrison-Woodbury
        # A_new = A_old + x x^T  =>  A_inv_new = A_inv_old - (A_inv_old x x^T A_inv_old) / (1 + x^T A_inv_old x)
        x = context
        Ax = self.A_inv[arm] @ x
        denom = 1.0 + x @ Ax
        if abs(denom) < 1e-12:
            denom = 1e-12
        self.A_inv[arm] = self.A_inv[arm] - np.outer(Ax, Ax) / denom
        self.b[arm] = self.b[arm] + reward * x
# EVOLVE-BLOCK-END
