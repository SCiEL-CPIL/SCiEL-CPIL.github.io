"""Non-stationary linear contextual bandit with arm-identity features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Single global linear model with arm-identity one-hot features.
    
    Extended context: [x, onehot(arm)] -> reward
    This allows the model to learn:
    - Context-dependent effects shared across arms
    - Arm-specific offsets/corrections
    - Inter-arm correlations through shared context weights
    
    Uses adaptive forgetting factor and UCB exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Extended dimension: original context + one-hot arm indicators
        self.ext_dim = dim + n_arms

        # Single global linear model
        # A: ext_dim x ext_dim
        # b: ext_dim vector
        # inv_A: ext_dim x ext_dim
        self.A = np.eye(self.ext_dim)
        self.b = np.zeros(self.ext_dim)
        self.inv_A = np.eye(self.ext_dim)

        # Adaptive forgetting
        self.base_lambda = 0.93
        self.min_lambda = 0.80
        self.max_lambda = 0.99
        self.error_buffer = deque(maxlen=50)

        # Adaptive UCB
        self.base_alpha = 1.2
        self.min_alpha = 0.4
        self.max_alpha = 3.0

        # Track per-arm counts for confidence weighting
        self.arm_counts = np.zeros(n_arms)

        # Time guard
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0

        # Round counter
        self._round = 0

        # Cache for quick prediction
        self._theta = np.zeros(self.ext_dim)

    def _extend_context(self, context, arm):
        """Create extended context vector with one-hot arm indicator."""
        ext = np.zeros(self.ext_dim)
        ext[:self.d] = context
        ext[self.d + arm] = 1.0
        return ext

    def _compute_lambda(self):
        """Adaptive forgetting factor based on recent prediction errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Robust scale estimate
        if len(recent) >= 10:
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + std_err * 0.5) / err_scale

        if normalized < 0.10:
            return self.max_lambda
        elif normalized > 2.0:
            return self.min_lambda
        else:
            t = (normalized - 0.10) / (2.0 - 0.10)
            t = t * t * (3.0 - 2.0 * t)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _compute_alpha(self):
        """Adaptive exploration parameter."""
        if len(self.error_buffer) < 10:
            return self.base_alpha

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        if len(recent) >= 10:
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + 0.5 * std_err) / err_scale

        if normalized < 0.15:
            return self.min_alpha
        elif normalized > 1.8:
            return self.max_alpha
        else:
            t = (normalized - 0.15) / (1.8 - 0.15)
            t = t * t * (3.0 - 2.0 * t)
            return self.min_alpha + t * (self.max_alpha - self.min_alpha)

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        # Update theta cache
        self._theta = self.inv_A @ self.b

        alpha = self._compute_alpha()

        # Compute scores for each arm using the single global model
        # For arm a: mu_a = theta^T [x, onehot(a)]
        #           = theta[:d]^T x + theta[d+a]
        scores = np.zeros(self.n)
        
        # Precompute context part
        context_score = self._theta[:self.d] @ context
        
        for a in range(self.n):
            # Mean prediction
            mu = context_score + self._theta[self.d + a]
            
            # Variance: need ext_context^T inv_A ext_context
            # ext_context = [x, onehot(a)]
            # var = x^T inv_A[:d,:d] x + 2 * x^T inv_A[:d, d+a] + inv_A[d+a, d+a]
            # But inv_A changes per arm selection, so we compute per arm
            # For efficiency, compute the variance components
            
            # inv_A is symmetric
            # var = [x, e_a]^T inv_A [x, e_a]
            #     = x^T inv_A[0:d, 0:d] x + 2*x^T inv_A[0:d, d+a] + inv_A[d+a, d+a]
            
            # Precompute x part
            # Actually, let's just compute directly for clarity and correctness
            # The dimension is small (d + n_arms), so this is fine
            
            # Build extended context for this arm
            ext_x = np.zeros(self.ext_dim)
            ext_x[:self.d] = context
            ext_x[self.d + a] = 1.0
            
            # Variance = ext_x^T inv_A ext_x
            inv_A_ext = self.inv_A @ ext_x
            var = ext_x @ inv_A_ext
            std = np.sqrt(max(0.0, var))
            
            # UCB score with confidence-weighted exploration
            count_factor = 1.0 / (1.0 + 0.05 * self.arm_counts[a])
            scores[a] = mu + alpha * std * count_factor

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self._deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error BEFORE updating
        # Reconstruct what the prediction would have been
        ext_x = np.zeros(self.ext_dim)
        ext_x[:self.d] = context
        ext_x[self.d + arm] = 1.0
        
        pred = self._theta @ ext_x
        error = abs(reward - pred)
        self.error_buffer.append(error)

        # Adaptive forgetting factor
        lam = self._compute_lambda()

        # Update the single global linear model with forgetting
        # A_new = lam * A_old + x x^T
        # b_new = lam * b_old + r * x
        
        A_old = self.A
        b_old = self.b
        
        A_new = lam * A_old + np.outer(ext_x, ext_x)
        b_new = lam * b_old + reward * ext_x

        # Update inverse via Sherman-Morrison:
        # (lam*A + xx^T)^{-1} = (1/lam) * (A^{-1} - A^{-1} x x^T A^{-1} / (lam + x^T A^{-1} x))
        inv_A_old = self.inv_A
        Ax = inv_A_old @ ext_x
        xAx = ext_x @ Ax
        denom = lam + xAx
        
        if abs(denom) > 1e-10:
            inv_A_new = (1.0 / lam) * (inv_A_old - np.outer(Ax, Ax) / denom)
            # Symmetrize to reduce numerical drift
            inv_A_new = 0.5 * (inv_A_new + inv_A_new.T)
        else:
            # Fallback: recompute with regularization
            try:
                inv_A_new = np.linalg.inv(A_new + 1e-6 * np.eye(self.ext_dim))
            except np.linalg.LinAlgError:
                inv_A_new = np.eye(self.ext_dim)

        # Store updates
        self.A = A_new
        self.b = b_new
        self.inv_A = inv_A_new
        self._theta = self.inv_A @ self.b

        # Update arm count
        self.arm_counts[arm] += 1

# EVOLVE-BLOCK-END
