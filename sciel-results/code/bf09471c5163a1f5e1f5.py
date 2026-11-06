"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with SMW inverse updates, adaptive forgetting, and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 115.0
        self._recent_errors = []
        self._base_lambda = 0.96
        self._min_lambda = 0.84
        self._max_lambda = 0.995
        self._alpha = 0.45
        self._round = 0
        self._fallback_arm = 0
        self._error_window = 20

    def _get_lambda(self):
        if len(self._recent_errors) < 10:
            return self._base_lambda
        recent = self._recent_errors[-self._error_window:]
        recent_mean_error = np.mean(recent)
        recent_std_error = np.std(recent) if len(recent) > 2 else 0.0
        # Combine mean and variability: high error OR high variability => faster forgetting
        error_signal = 0.7 * recent_mean_error + 0.3 * recent_std_error
        error_norm = np.clip(error_signal / 1.5, 0.0, 1.0)
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
                theta = self.A_inv[a] @ self.b[a]
                score = float(theta @ context)
                if score > best_score:
                    best_score = score
                    best_arm = a
            self._fallback_arm = best_arm
            return int(best_arm)

        # Adaptive alpha: reduce exploration as confidence grows
        # Use average inverse diagonal as a proxy for confidence
        avg_confidence = np.mean([np.trace(self.A_inv[a]) for a in range(self.n)])
        # Scale alpha down when confidence is high (low trace of A_inv)
        alpha_eff = self._alpha * np.clip(1.0 / (1.0 + avg_confidence * 0.1), 0.5, 1.0)

        scores = np.empty(self.n)
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            pred = float(theta @ context)
            var = float(context @ (inv @ context))
            if var < 0.0:
                var = 0.0
            scores[a] = pred + alpha_eff * np.sqrt(var)
        arm = int(np.argmax(scores))
        self._fallback_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        theta = self.A_inv[arm] @ self.b[arm]
        pred = float(theta @ context)
        error = abs(reward - pred)
        self._recent_errors.append(error)
        if len(self._recent_errors) > 50:
            self._recent_errors.pop(0)

        lam = self._get_lambda()
        self._round += 1

        # SMW: A_new = lam * A_old + x x^T
        # (lam * A_old)^{-1} = A_old^{-1} / lam
        # Then (lam*A_old + x x^T)^{-1} via SMW
        M_inv = self.A_inv[arm] / lam
        M_inv_x = M_inv @ context
        denom = 1.0 + float(context @ M_inv_x)
        if abs(denom) < 1e-12:
            denom = 1e-12
        self.A_inv[arm] = M_inv - np.outer(M_inv_x, M_inv_x) / denom

        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        if self._round % 200 == 0:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                pass
# EVOLVE-BLOCK-END
