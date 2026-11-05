# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive non-stationary linear bandit with error-driven forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.t0 = time.time()

        # Per-arm: maintain inverse covariance V and mean vector b
        # theta = V @ b  (since A^{-1} = V, and A theta = b => theta = V b)
        # Use a slightly smaller initial ridge to allow faster learning, but regularize later
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting parameters (Tuned based on knowledge: gamma=0.9, range [0.85, 0.99])
        self.base_gamma = 0.9
        self.gamma_min = 0.85
        self.gamma_max = 0.99
        self.gamma = [self.base_gamma for _ in range(n_arms)]
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.pred = [0.0 for _ in range(n_arms)]

        # Exploration
        self.C = 1.3
        self.explore_decay = 1000.0

        # Track best arm for deadline safety
        self.best_arm = 0
        self.best_score = -1e9

    def _elapsed(self):
        return time.time() - self.t0

    def _compute_gamma(self, arm, error):
        """Adaptively adjust gamma. Faster adaptation (0.2 weight) and wider range."""
        # Update EMA with faster weight (0.2) as per knowledge
        self.error_ema[arm] = 0.8 * self.error_ema[arm] + 0.2 * error
        # Map error to gamma reduction. High error -> lower gamma (more forgetting).
        # Normalize error to [0, 1] range roughly.
        norm_error = min(1.0, self.error_ema[arm] / 10.0) 
        reduction = norm_error * (self.base_gamma - self.gamma_min)
        new_gamma = self.base_gamma - reduction
        # Clamp to [gamma_min, gamma_max]
        self.gamma[arm] = max(self.gamma_min, min(self.gamma_max, new_gamma))
        # Slowly pull back to base if error is low
        if self.error_ema[arm] < 1.0:
            self.gamma[arm] = 0.95 * self.gamma[arm] + 0.05 * self.base_gamma
        return self.gamma[arm]

    def select(self, context):
        # Deadline guard: if close to limit, return best known arm
        if self._elapsed() > 145.0:
            return self.best_arm

        self.t += 1
        ctx = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        try:
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                mu = theta @ ctx
                # Exploration bonus with slow decay
                vctx = self.V[a] @ ctx
                var = ctx @ vctx
                if var < 0:
                    var = 0.0
                bonus = self.C * np.sqrt(var)
                bonus *= (1.0 / (1.0 + self.t / self.explore_decay))
                scores[a] = mu + bonus

            arm = int(np.argmax(scores))
            if scores[arm] > self.best_score:
                self.best_score = scores[arm]
                self.best_arm = arm
        except Exception:
            # If numerical issues occur, fall back to best arm or random
            arm = self.best_arm
            # Reset V and b for that arm to identity/zero to recover
            self.V[self.best_arm] = np.eye(self.d)
            self.b[self.best_arm] = np.zeros(self.d)

        return arm

    def update(self, context, arm, reward):
        if self._elapsed() > 145.0:
            return

        ctx = np.asarray(context, dtype=np.float64)
        r = float(reward)
        a = int(arm)

        try:
            # Compute prediction for adaptive gamma
            theta = self.V[a] @ self.b[a]
            pred = theta @ ctx
            self.pred[a] = pred
            error = (r - pred) ** 2
            gamma = self._compute_gamma(a, error)

            # Sherman-Morrison update for inverse covariance with forgetting
            V = self.V[a]
            Vctx = V @ ctx
            ctxVctx = ctx @ Vctx
            
            # Ensure positive definiteness in the denominator
            if ctxVctx < 0:
                ctxVctx = 0.0
            
            denom = 1.0 / gamma + ctxVctx
            if denom > 1e-12:
                # (1/gamma) * V - (1/gamma^2) * outer(Vctx, Vctx) / denom
                # Note: Vctx = V @ ctx
                coeff = (1.0 / (gamma * gamma)) / denom
                V_new = (1.0 / gamma) * V - coeff * np.outer(Vctx, Vctx)
            else:
                V_new = (1.0 / gamma) * V

            # Symmetrize for numerical stability
            V_new = 0.5 * (V_new + V_new.T)
            
            # Check for NaNs or Infs
            if not np.all(np.isfinite(V_new)):
                V_new = np.eye(self.d)
            
            self.V[a] = V_new
            self.b[a] = gamma * self.b[a] + r * ctx

            # Periodic ridge to prevent numerical drift
            if self.t % 100 == 0:
                for i in range(self.n):
                    self.V[i] += 1e-3 * np.eye(self.d)
        except Exception:
            # Fallback: reset the arm's model to initial state
            self.V[a] = np.eye(self.d)
            self.b[a] = np.zeros(self.d)
            self.gamma[a] = self.base_gamma
            self.error_ema[a] = 0.0
# EVOLVE-BLOCK-END
