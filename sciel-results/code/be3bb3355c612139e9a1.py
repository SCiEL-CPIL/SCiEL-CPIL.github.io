"""Non-stationary linear contextual bandit with adaptive forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    LinUCB with SMW (Sherman-Morrison-Woodbury) inverse updates,
    adaptive forgetting factor, and adaptive exploration.
    
    Uses per-arm linear models with a forgetting mechanism that adapts
    based on prediction error. Higher error increases the forgetting rate
    to track non-stationarity faster, and increases exploration.
    Maintains A_inv explicitly for O(d^2) updates instead of O(d^3) inversions.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm ridge regression matrices and their inverses
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive parameters
        self.base_forgetting = 0.95  # Base forgetting factor (faster base)
        self.forgetting = self.base_forgetting
        self.base_alpha = 1.5        # Base exploration parameter
        self.alpha = self.base_alpha
        # Error tracking
        self.recent_errors = []
        self.error_window = 30
        # Per-arm visit counts for count-based exploration decay
        self.arm_counts = [0] * n_arms
        # Time tracking for deadline safety
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill
        # Track best arm per context pattern for fallback
        self.arm_rewards = {a: [] for a in range(n_arms)}

    def _compute_adaptive_params(self):
        """Compute adaptive forgetting and alpha based on recent prediction errors."""
        if len(self.recent_errors) >= 10:
            recent = self.recent_errors[-self.error_window:]
            mean_err = np.mean(np.abs(recent))
            # Scale forgetting: higher error -> more forgetting (closer to 0)
            # Clamp between 0.78 and 0.999 (faster forgetting range)
            self.forgetting = np.clip(0.999 - 0.08 * mean_err, 0.78, 0.999)
            # Scale alpha: higher error -> more exploration
            self.alpha = np.clip(self.base_alpha + 0.8 * mean_err, 0.5, 4.0)
        else:
            self.forgetting = self.base_forgetting
            self.alpha = self.base_alpha

    def _is_deadline_near(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: if we're running low on time, use a simple heuristic
        if self._is_deadline_near():
            # Pick the arm with highest average reward so far
            if any(len(r) > 0 for r in self.arm_rewards.values()):
                avg_rewards = [np.mean(r) if r else 0.0 for r in self.arm_rewards.values()]
                return int(np.argmax(avg_rewards))
            return 0

        self._compute_adaptive_params()

        scores = []
        for a in range(self.n):
            try:
                theta = self.A_inv[a] @ self.b[a]
                mean_reward = theta @ context
                variance = context @ self.A_inv[a] @ context
                # Count-based exploration decay: reduce exploration for well-sampled arms
                count_factor = 1.0 / (1.0 + self.arm_counts[a] * 0.01)
                score = mean_reward + self.alpha * count_factor * np.sqrt(max(variance, 0))
            except np.linalg.LinAlgError:
                score = 0.0
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction error for adaptive params
        try:
            theta = self.A_inv[arm] @ self.b[arm]
            predicted = theta @ context
            error = abs(reward - predicted)
        except np.linalg.LinAlgError:
            error = abs(reward)

        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        # Update arm count
        self.arm_counts[arm] += 1
        
        # Update the arm's model with forgetting using SMW
        # New A = forgetting * A_old + outer(context, context)
        # We maintain A_inv explicitly.
        # A_new = f * A_old + x x^T
        # A_new^{-1} = (f * A_old + x x^T)^{-1}
        # = (f * (A_old + (1/f) x x^T))^{-1}
        # = (1/f) * (A_old + (1/f) x x^T)^{-1}
        # Using SMW on (A_old + (1/f) x x^T):
        # (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here u = x/sqrt(f), v = x/sqrt(f), so u v^T = (1/f) x x^T
        # (A_old + (1/f) x x^T)^{-1} = A_inv - A_inv (x/sqrt(f)) (x/sqrt(f))^T A_inv / (1 + (1/f) x^T A_inv x)
        # = A_inv - (1/f) A_inv x x^T A_inv / (1 + (1/f) x^T A_inv x)
        # = A_inv - A_inv x x^T A_inv / (f + x^T A_inv x)
        
        f = self.forgetting
        A_inv = self.A_inv[arm]
        
        # Compute intermediate values
        Ax = A_inv @ context
        xA = context @ A_inv  # This is the transpose of Ax if A_inv is symmetric, but let's be explicit
        # Actually x^T A_inv x is a scalar
        xAx = context @ Ax
        
        # SMW update for A_inv
        # A_new_inv = (1/f) * (A_inv - A_inv x x^T A_inv / (f + x^T A_inv x))
        # Wait, let me re-derive carefully.
        # A_new = f * A_old + x x^T
        # A_new^{-1} = (f * A_old + x x^T)^{-1}
        # = [f (A_old + (1/f) x x^T)]^{-1}
        # = (1/f) [A_old + (1/f) x x^T]^{-1}
        # Let B = A_old, C = (1/f) x x^T
        # (B + C)^{-1} = B^{-1} - B^{-1} C (I + C B^{-1})^{-1}
        # Or using SMW: (B + u v^T)^{-1} = B^{-1} - B^{-1} u (I + v^T B^{-1} u)^{-1} v^T B^{-1}
        # Here u = x/sqrt(f), v = x/sqrt(f)
        # (I + v^T B^{-1} u) = 1 + (1/f) x^T B^{-1} x = 1 + xAx/f = (f + xAx)/f
        # So (B + C)^{-1} = B^{-1} - B^{-1} (x/sqrt(f)) ((f + xAx)/f)^{-1} (x/sqrt(f))^T B^{-1}
        # = B^{-1} - B^{-1} x x^T B^{-1} * (1/f) * (f/(f + xAx))
        # = B^{-1} - B^{-1} x x^T B^{-1} / (f + xAx)
        # Therefore A_new^{-1} = (1/f) * (B^{-1} - B^{-1} x x^T B^{-1} / (f + xAx))
        # = (1/f) B^{-1} - (1/f) B^{-1} x x^T B^{-1} / (f + xAx)
        
        denom = f + xAx
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse from A
            self.A[arm] = f * self.A[arm] + np.outer(context, context)
            self.b[arm] = f * self.b[arm] + reward * context
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d)
        else:
            # Update A_inv using SMW
            A_inv_x = A_inv @ context
            A_inv_x_xT_A_inv = np.outer(A_inv_x, A_inv_x)
            A_inv_new = (1.0 / f) * A_inv - (1.0 / (f * denom)) * A_inv_x_xT_A_inv
            self.A_inv[arm] = A_inv_new
            
            # Update A (for potential debugging or fallback)
            self.A[arm] = f * self.A[arm] + np.outer(context, context)
            
            # Update b
            self.b[arm] = f * self.b[arm] + reward * context

        # Track rewards for fallback
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > 100:
            self.arm_rewards[arm].pop(0)
# EVOLVE-BLOCK-END
