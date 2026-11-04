"""
Rewritten program for non-stationary linear contextual bandit.
Uses Sliding Window Linear Regression with adaptive window size.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Sliding window parameters
        self.window_size = 50
        self.min_window = 10
        self.max_window = 200
        
        # Per-arm data structures
        # We store the last W (x, r) pairs for each arm to allow removal of old data
        self.queues = [[] for _ in range(n_arms)]
        
        # Current V and b for fast prediction
        # V = (sum x x^T + lambda I)^-1
        # b = sum r x
        self.lam = 1.0  # Ridge regularization
        self.V = [np.eye(dim) / self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive window control
        self.residual_ema = np.zeros(n_arms)
        self.residual_alpha = 0.1
        
        # Exploration
        self.alpha = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.5

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 118.0
        self.round = 0
        self.best_arm = 0
        self.best_reward = -np.inf
        self.arm_mean_reward = [0.0] * n_arms
        self.arm_count = [0] * n_arms

        # Re-inversion counter for numerical stability
        self.reinv_counter = [0] * n_arms
        self.reinv_interval = 100

    def _check_deadline(self):
        if time.time() - self.start_time >= self.deadline:
            return True
        return False

    def _update_window_size(self, arm, residual):
        """Adapt window size based on residual magnitude."""
        self.residual_ema[arm] = (1 - self.residual_alpha) * self.residual_ema[arm] + \
                                 self.residual_alpha * abs(residual)
        
        r = self.residual_ema[arm]
        
        # If residual is high, shrink window to forget faster
        # If residual is low, grow window to use more data
        if r > 2.0:
            self.window_size = max(self.min_window, int(self.window_size * 0.95))
        elif r < 0.5:
            self.window_size = min(self.max_window, int(self.window_size * 1.05))
        else:
            # Keep current size or slight drift towards middle
            target = 50 + (r - 1.0) * 20
            self.window_size = int(self.window_size * 0.9 + target * 0.1)
            self.window_size = max(self.min_window, min(self.max_window, self.window_size))

    def _update_adaptive_alpha(self):
        """Adapt exploration coefficient."""
        # Start with high exploration, decay over time
        if self.round < 50:
            self.alpha = self.alpha_max
        elif self.round < 200:
            self.alpha = self.alpha_max - (self.alpha_max - 1.0) * (self.round - 50) / 150.0
        else:
            self.alpha = max(self.alpha_min, 1.0 - 0.2 * (self.round - 200) / 1000.0)

    def _add_to_queue(self, arm, context, reward):
        """Add new observation and update V, b using Sherman-Morrison."""
        q = self.queues[arm]
        q.append((context, reward))
        
        # If queue exceeds window size, remove oldest
        if len(q) > self.window_size:
            old_ctx, old_rew = q.pop(0)
            self._remove_sample(arm, old_ctx, old_rew)
        
        self._add_sample(arm, context, reward)

    def _add_sample(self, arm, x, r):
        """Add a sample to the linear model."""
        V = self.V[arm]
        b = self.b[arm]
        
        # Sherman-Morrison for A += x x^T
        # V_new = V - V x x^T V / (1 + x^T V x)
        Vx = V @ x
        denom = 1.0 + x @ Vx
        if denom < 1e-10:
            denom = 1e-10
        self.V[arm] = V - np.outer(Vx, Vx) / denom
        self.b[arm] = b + r * x
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def _remove_sample(self, arm, x, r):
        """Remove a sample from the linear model (inverse Sherman-Morrison)."""
        V = self.V[arm]
        b = self.b[arm]
        
        # We are removing x x^T from A.
        # V_old = (A - x x^T + x x^T)^-1 ? No.
        # We have V_current = A_current^-1.
        # We want V_new = (A_current - x x^T)^-1.
        # Let A' = A_current - x x^T.
        # V_new = (A_current - x x^T)^-1
        # Using matrix inversion lemma: (A - u v^T)^-1 = A^-1 + A^-1 u v^T A^-1 / (1 - v^T A^-1 u)
        # Here u=x, v=x.
        # V_new = V + V x x^T V / (1 - x^T V x)
        
        Vx = V @ x
        denom = 1.0 - x @ Vx
        
        # If denom is close to 0, the removal is numerically unstable or the sample was dominant.
        # In that case, we might need to re-invert from scratch later.
        if abs(denom) < 1e-8:
            # Fallback: mark for re-inversion, but for now, just skip update to avoid crash
            # or do a small perturbation
            denom = 1e-8 if denom > 0 else -1e-8
            
        self.V[arm] = V + np.outer(Vx, Vx) / denom
        self.b[arm] = b - r * x
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def _reinvert(self, arm):
        """Recompute V from scratch to clean numerical errors."""
        q = self.queues[arm]
        if not q:
            self.V[arm] = np.eye(self.dim) / self.lam
            self.b[arm] = np.zeros(self.dim)
            self.theta[arm] = np.zeros(self.dim)
            return
        
        A = self.lam * np.eye(self.dim)
        b = np.zeros(self.dim)
        for x, r in q:
            A += np.outer(x, x)
            b += r * x
            
        try:
            self.V[arm] = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.V[arm] = np.eye(self.dim) / self.lam
        
        self.b[arm] = b
        self.theta[arm] = self.V[arm] @ b

    def select(self, context):
        self.round += 1

        if self._check_deadline():
            return self.best_arm

        context = np.asarray(context, dtype=np.float64).flatten()

        self._update_adaptive_alpha()

        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Uncertainty: context^T V_a context
            var = context @ self.V[a] @ context
            # Ensure var is positive
            if var < 0:
                var = 0.0
            scores[a] = mean + self.alpha * np.sqrt(var)

        # Small random tiebreaker
        scores += np.random.normal(0, 0.01, self.n)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)

        # Track for fallback
        self.arm_count[arm] += 1
        # Simple mean tracking for best_arm
        if self.arm_count[arm] == 1:
            self.arm_mean_reward[arm] = reward
        else:
            self.arm_mean_reward[arm] = (self.arm_mean_reward[arm] * (self.arm_count[arm] - 1) + reward) / self.arm_count[arm]
        
        if self.arm_count[arm] >= 5:
            if self.arm_mean_reward[arm] > self.best_reward:
                self.best_reward = self.arm_mean_reward[arm]
                self.best_arm = arm

        # Prediction before update
        pred = self.theta[arm] @ context
        residual = reward - pred

        # Adapt window size
        self._update_window_size(arm, residual)

        # Add new sample
        self._add_to_queue(arm, context, reward)

        # Periodic re-inversion for stability
        self.reinv_counter[arm] += 1
        if self.reinv_counter[arm] >= self.reinv_interval:
            self._reinvert(arm)
            self.reinv_counter[arm] = 0
# EVOLVE-BLOCK-END
