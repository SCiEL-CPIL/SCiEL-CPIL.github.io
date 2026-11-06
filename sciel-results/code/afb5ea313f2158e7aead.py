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
        self._deadline = self._start_time + 135.0
        self._recent_errors = []
        self._ema_error = 0.0
        self._ema_alpha = 0.1
        self._base_lambda = 0.96
        self._min_lambda = 0.88
        self._max_lambda = 0.99
        self._base_alpha = 0.5
        self._min_alpha = 0.3
        self._max_alpha = 0.8
        self._round = 0
        self._fallback_arm = 0
        self._context_mean = np.zeros(dim)
        self._context_std = np.ones(dim)
        self._n_obs = 0

    def _get_lambda(self):
        # Smooth adaptive lambda using logistic mapping on EMA error
        # Higher error -> lower lambda (faster forgetting)
        if self._ema_error < 0.1:
            error_norm = 0.0
        elif self._ema_error > 3.0:
            error_norm = 1.0
        else:
            # Logistic smooth mapping from [0.1, 3.0] to [0, 1]
            z = (self._ema_error - 1.55) / 0.75
            error_norm = 1.0 / (1.0 + np.exp(-z))
        
        lam = self._max_lambda - error_norm * (self._max_lambda - self._min_lambda)
        return float(lam)

    def _get_alpha(self):
        # Smooth adaptive alpha using logistic mapping
        # Higher error -> higher alpha (more exploration)
        if self._ema_error < 0.1:
            error_norm = 0.0
        elif self._ema_error > 3.0:
            error_norm = 1.0
        else:
            z = (self._ema_error - 1.55) / 0.75
            error_norm = 1.0 / (1.0 + np.exp(-z))
        
        alpha = self._min_alpha + error_norm * (self._max_alpha - self._min_alpha)
        return float(alpha)

    def _normalize_context(self, context):
        # Normalize context using running mean/std for better numerical properties
        if self._n_obs < 10:
            return context
        normed = (context - self._context_mean) / (self._context_std + 1e-8)
        return normed

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

        # Use normalized context for selection
        ctx_norm = self._normalize_context(context)
        
        scores = np.empty(self.n)
        alpha = self._get_alpha()
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            pred = float(theta @ ctx_norm)
            var = float(ctx_norm @ (inv @ ctx_norm))
            if var < 0.0:
                var = 0.0
            scores[a] = pred + alpha * np.sqrt(var)
        arm = int(np.argmax(scores))
        self._fallback_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        ctx_norm = self._normalize_context(context)
        
        # Update running statistics for context normalization
        self._n_obs += 1
        if self._n_obs <= 1000:
            # Online mean/variance update
            delta = context - self._context_mean
            self._context_mean += delta / self._n_obs
            self._context_std = np.sqrt(self._context_std**2 + (delta**2 - self._context_std**2) / self._n_obs)
            self._context_std = np.maximum(self._context_std, 0.1)
        
        theta = self.A_inv[arm] @ self.b[arm]
        pred = float(theta @ ctx_norm)
        error = abs(reward - pred)
        
        # Update EMA error for smooth adaptation
        self._ema_error = (1 - self._ema_alpha) * self._ema_error + self._ema_alpha * error
        self._recent_errors.append(error)
        if len(self._recent_errors) > 50:
            self._recent_errors.pop(0)

        lam = self._get_lambda()
        self._round += 1

        # SMW: A_new = lam * A_old + x x^T
        # (lam * A_old)^{-1} = A_old^{-1} / lam
        # Then (lam*A_old + x x^T)^{-1} via SMW
        M_inv = self.A_inv[arm] / lam
        M_inv_x = M_inv @ ctx_norm
        denom = 1.0 + float(ctx_norm @ M_inv_x)
        if abs(denom) < 1e-12:
            denom = 1e-12
        self.A_inv[arm] = M_inv - np.outer(M_inv_x, M_inv_x) / denom

        self.A[arm] = lam * self.A[arm] + np.outer(ctx_norm, ctx_norm)
        self.b[arm] = lam * self.b[arm] + reward * ctx_norm

        if self._round % 500 == 0:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                pass
# EVOLVE-BLOCK-END
