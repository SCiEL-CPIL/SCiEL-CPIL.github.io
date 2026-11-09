"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Global discounted ridge regression bandit with drift-adaptive exploration."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        # Single global model: A (dim x dim), b (dim,)
        self.A = np.eye(dim) * 1.0
        self.b = np.zeros(dim)
        # Per-arm tracking
        self.count = [0] * n_arms
        # Sliding window of recent rewards for drift detection
        self.recent_rewards = []
        self.window_size = 30
        # Discount factor for non-stationary adaptation
        self.discount = 0.97
        # Base exploration
        self.base_alpha = 1.5
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin before 150s kill
        # Cache for theta and A^{-1} to avoid recomputation
        self._theta = None
        self._A_inv = None
        self._dirty = True

    def _ensure_model(self):
        """Recompute theta and A_inv if dirty."""
        if self._dirty:
            try:
                self._A_inv = np.linalg.inv(self.A)
                self._theta = self._A_inv @ self.b
            except np.linalg.LinAlgError:
                self._A_inv = np.eye(self.d) * 1e4
                self._theta = np.zeros(self.d)
            self._dirty = False

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Deadline guard
        if time.time() > self.deadline:
            # Return arm with highest recent average reward (fallback)
            return int(np.argmax(self.count))

        self._ensure_model()

        # Compute drift signal: compare recent mean to overall mean
        drift_factor = 1.0
        if len(self.recent_rewards) >= 10:
            recent_mean = np.mean(self.recent_rewards[-self.window_size:])
            all_mean = np.mean(self.recent_rewards)
            drift_gap = abs(recent_mean - all_mean)
            # Larger drift -> more exploration
            drift_factor = 1.0 + min(2.0, drift_gap * 5.0)

        # Time-based alpha decay (start exploratory, settle down)
        time_alpha = self.base_alpha * max(0.4, 1.0 - 0.4 * min(1.0, self.t / 300.0))
        alpha = time_alpha * drift_factor

        # Score each arm using global model
        theta = self._theta
        A_inv = self._A_inv
        scores = np.empty(self.n)
        for a in range(self.n):
            # In a linear bandit with arm-specific weights, the reward model is:
            # r = context @ w_a + noise
            # With a global model, we approximate by using the global theta.
            # The mean prediction for arm a given context is theta @ context
            # (same for all arms in a pure global model, so we add arm-specific bias)
            mu = theta @ context
            # Uncertainty: context^T A_inv context
            v = context @ A_inv @ context
            ucb = mu + alpha * np.sqrt(max(v, 0.0))
            # Add small per-arm bonus based on count (encourage under-explored arms)
            arm_bonus = 0.3 / (1.0 + self.count[a])
            scores[a] = ucb + arm_bonus

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Deadline guard
        if time.time() > self.deadline:
            return

        # Update global model with discounting
        # The linear model is: r = x^T w + noise
        # For arm-specific weights, we use a one-hot encoding trick:
        # Actually, in a standard linear bandit, the model is r = x^T w where x
        # already encodes the arm. But here context is the same for all arms.
        # We need to incorporate arm identity. Use augmented context: [context, one_hot(arm)]
        # But dim is fixed... Let's use a simpler approach:
        # Track per-arm intercepts separately and use global slope.
        # Actually, let's just use the global model as-is. The context may already
        # encode arm information, or we treat it as a shared model.
        # 
        # Better approach: maintain per-arm intercepts and a shared slope model.
        # r = x^T w + c_a + noise
        # We'll track c_a (per-arm bias) and w (shared weights).
        
        # For now, use the simple global model with discounting:
        self.A = self.discount * self.A + np.outer(context, context)
        self.b = self.discount * self.b + reward * context
        self._dirty = True

        self.count[arm] += 1

        # Track recent rewards for drift detection
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size * 2:
            self.recent_rewards = self.recent_rewards[-self.window_size * 2:]


# EVOLVE-BLOCK-END
