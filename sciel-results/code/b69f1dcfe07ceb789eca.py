"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive LinUCB with Sherman-Morrison inverse-covariance updates,
    adaptive forgetting factor, and deadline guard for non-stationary
    linear contextual bandits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = self._t0 + 115.0  # safety margin before 120s kill

        # Store V = (A)^{-1} for efficient Sherman-Morrison updates
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.97
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.base_C = 1.5
        self.C_min = 0.5
        self.C_max = 3.0

        # Track recent rewards for adaptive tuning
        self.reward_history = []
        self.max_history = 100

        # Per-arm recent reward tracking for adaptive forgetting
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_reward_counts = [0] * n_arms

    def _check_deadline(self):
        """Return True if we should stop exploring and just pick best known arm."""
        return time.time() > self._deadline

    def _adaptive_gamma(self):
        """Adapt forgetting factor based on reward drift (change in mean)."""
        if len(self.reward_history) < 20:
            return self.base_gamma

        # Compare recent half vs earlier half to detect drift
        n = len(self.reward_history)
        half = n // 2
        recent_mean = np.mean(self.reward_history[-half:])
        earlier_mean = np.mean(self.reward_history[:half])
        
        # Scale by typical reward magnitude
        scale = np.mean(np.abs(self.reward_history)) + 1e-9
        drift = abs(recent_mean - earlier_mean) / scale

        # Low drift -> high gamma (keep more history), high drift -> low gamma (forget fast)
        # Map drift in [0, ~1] to gamma in [gamma_max, gamma_min]
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * min(1.0, drift * 3.0)
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration parameter with smooth decay and per-arm confidence."""
        total_count = sum(self.counts)
        if total_count < 5:
            return self.C_max

        # Smooth exponential decay of exploration
        progress = min(1.0, total_count / 300.0)
        C = self.C_max * np.exp(-2.0 * progress)

        # Boost exploration for under-explored arms
        if self.counts[arm] < total_count / self.n:
            C *= 1.5

        return max(self.C_min, min(self.C_max, C))

    def select(self, context):
        if self._check_deadline():
            # Deadline approaching: pick the arm with highest estimated reward
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # Uncertainty bonus with numerical safeguard
            v = self.V[a]
            var = context @ v @ context
            if var < 0:
                var = 0.0
            uncertainty = np.sqrt(var)
            C = self._adaptive_C(a)
            scores.append(mu + C * uncertainty)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        gamma = self._adaptive_gamma()

        V = self.V[arm]
        x = context
        gamma_inv = 1.0 / gamma

        Vx = V @ x
        xVx = x @ Vx  # scalar

        denom = 1.0 + gamma_inv * xVx

        # Sherman-Morrison with regularization for numerical stability
        self.V[arm] = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom

        # Add small regularization to prevent singularity
        self.V[arm] += 1e-8 * np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * x

        self.counts[arm] += 1

        # Track reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        # Track per-arm reward stats
        self.arm_reward_sums[arm] = gamma * self.arm_reward_sums[arm] + reward
        self.arm_reward_counts[arm] = gamma * self.arm_reward_counts[arm] + 1
# EVOLVE-BLOCK-END
