"""Adaptive linear bandit policy for non-stationary settings."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Per-arm linear bandit with adaptive forgetting.
    
    Model: reward = x^T theta_a + noise
    - Each arm has its own linear parameter vector theta_a.
    - Adaptive forgetting factor reacts to prediction errors.
    - Adaptive UCB alpha based on recent error magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Hybrid model: global linear + arm-specific bias
        # Global model: A_g, b_g, inv_A_g (dim x dim, dim,)
        # Arm-specific biases: bias[a] (scalar)
        # Per-arm residual models for robustness: A[a], b[a], inv_A[a]
        
        self.A_g = np.eye(dim)
        self.b_g = np.zeros(dim)
        self.inv_A_g = np.eye(dim)
        
        self.bias = np.zeros(n_arms)
        
        # Per-arm residual models (for robustness and adaptation)
        self.A = np.array([np.eye(dim) for _ in range(n_arms)])
        self.b = np.array([np.zeros(dim) for _ in range(n_arms)])
        self.inv_A = np.array([np.eye(dim) for _ in range(n_arms)])
        
        # Mixing weight: how much to rely on global vs per-arm
        self.mix_weight = 0.7  # 0.7 global, 0.3 per-arm
        
        # Adaptive forgetting
        self.base_lambda = 0.93
        self.min_lambda = 0.80
        self.max_lambda = 0.99
        self.error_buffer = deque(maxlen=60)
        
        # Adaptive UCB
        self.base_alpha = 1.2
        self.min_alpha = 0.4
        self.max_alpha = 3.0

        # Track predictions for error computation
        self._last_mu = np.zeros(n_arms)

        # Time guard
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0

        # Round counter
        self._round = 0

    def _compute_lambda(self):
        """Adaptive forgetting factor based on recent prediction errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Robust IQR-based error scaling
        q75, q25 = np.percentile(recent, [75, 25])
        iqr = max(q75 - q25, 1e-6)
        err_scale = max(iqr, 1e-6)
        
        # Normalize errors to determine forgetting rate
        # High error -> low lambda (forget fast)
        # Low error -> high lambda (remember)
        normalized = (mean_err + std_err * 0.5) / err_scale

        # Smoothstep interpolation
        if normalized < 0.1:
            return self.max_lambda
        elif normalized > 2.0:
            return self.min_lambda
        else:
            t = (normalized - 0.1) / (2.0 - 0.1)
            t = t * t * (3 - 2 * t)  # smoothstep
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _compute_alpha(self):
        """Adaptive exploration parameter."""
        if len(self.error_buffer) < 10:
            return self.base_alpha

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)

        # Robust IQR-based error scaling
        q75, q25 = np.percentile(recent, [75, 25])
        iqr = max(q75 - q25, 1e-6)
        err_scale = max(iqr, 1e-6)
        
        # When errors are high, explore more; when low, exploit
        normalized = mean_err / err_scale

        if normalized < 0.2:
            return self.min_alpha
        elif normalized > 1.5:
            return self.max_alpha
        else:
            t = (normalized - 0.2) / (1.5 - 0.2)
            t = t * t * (3 - 2 * t)  # smoothstep
            return self.min_alpha + t * (self.max_alpha - self.min_alpha)

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        # Get current adaptive parameters
        alpha = self._compute_alpha()

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Global model prediction
            theta_g = self.inv_A_g @ self.b_g
            mu_g = theta_g @ context
            var_g = context @ (self.inv_A_g @ context)
            
            # Per-arm residual prediction
            theta_a = self.inv_A[a] @ self.b[a]
            mu_a = theta_a @ context
            var_a = context @ (self.inv_A[a] @ context)
            
            # Combine: weighted average of global and per-arm
            w = self.mix_weight
            mu = w * mu_g + (1 - w) * (mu_a + self.bias[a])
            var = w * w * var_g + (1 - w) * (1 - w) * var_a
            std = np.sqrt(max(0.0, var))

            # UCB
            scores[a] = mu + alpha * std
            self._last_mu[a] = mu

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self._deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error BEFORE updating
        pred = self._last_mu[arm]
        error = abs(reward - pred)
        self.error_buffer.append(error)

        # Adaptive forgetting factor
        lam = self._compute_lambda()

        # Update global model with forgetting
        # A_g_new = lam * A_g_old + x x^T
        # b_g_new = lam * b_g_old + r * x
        A_g_old = self.A_g
        b_g_old = self.b_g
        
        A_g_new = lam * A_g_old + np.outer(context, context)
        b_g_new = lam * b_g_old + reward * context

        # Update global inverse via Sherman-Morrison
        inv_A_g_old = self.inv_A_g
        Ax_g = inv_A_g_old @ context
        xAx_g = context @ Ax_g
        denom_g = lam + xAx_g
        
        if abs(denom_g) > 1e-12:
            inv_A_g_new = (1.0 / lam) * (inv_A_g_old - np.outer(Ax_g, Ax_g) / denom_g)
            # Symmetrize to reduce numerical drift
            inv_A_g_new = 0.5 * (inv_A_g_new + inv_A_g_new.T)
        else:
            # Fallback: recompute
            try:
                inv_A_g_new = np.linalg.inv(A_g_new)
            except np.linalg.LinAlgError:
                inv_A_g_new = np.eye(self.d)

        # Update arm-specific bias
        # The bias captures the difference between the actual reward and the global model prediction
        theta_g = self.inv_A_g @ self.b_g
        mu_g = theta_g @ context
        residual = reward - mu_g
        # Exponentially weighted bias
        self.bias[arm] = lam * self.bias[arm] + (1 - lam) * residual

        # Update per-arm residual model with forgetting
        A_old = self.A[arm]
        b_old = self.b[arm]
        
        A_new = lam * A_old + np.outer(context, context)
        b_new = lam * b_old + residual * context

        # Update per-arm inverse via Sherman-Morrison
        inv_A_old = self.inv_A[arm]
        Ax = inv_A_old @ context
        xAx = context @ Ax
        denom = lam + xAx
        
        if abs(denom) > 1e-12:
            inv_A_new = (1.0 / lam) * (inv_A_old - np.outer(Ax, Ax) / denom)
            # Symmetrize to reduce numerical drift
            inv_A_new = 0.5 * (inv_A_new + inv_A_new.T)
        else:
            # Fallback: recompute
            try:
                inv_A_new = np.linalg.inv(A_new)
            except np.linalg.LinAlgError:
                inv_A_new = np.eye(self.d)

        # Store updates
        self.A_g = A_g_new
        self.b_g = b_g_new
        self.inv_A_g = inv_A_g_new
        self.A[arm] = A_new
        self.b[arm] = b_new
        self.inv_A[arm] = inv_A_new

# EVOLVE-BLOCK-END
