"""RLS contextual bandit with context-adaptive forgetting + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Per-arm RLS with context-adaptive forgetting factor and UCB."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 120.0
        # Per-arm RLS state: P = (A^{-1}), theta
        self.P = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting parameters
        self.lambda_base = 0.99
        self.gamma = 0.0005        # context-norm scaling for forgetting
        self.lam_min = 0.90
        self.lam_max = 0.999
        # UCB exploration
        self.alpha = 1.2
        # Adaptive process-noise / innovation tracking
        self.innov_ema = [0.0 for _ in range(n_arms)]
        self.innov_alpha = 0.9
        # Reward-scale normalization
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_scale_ema = [1.0 for _ in range(n_arms)]
        self.r_alpha = 0.95
        # Counters
        self.counts = [0 for _ in range(n_arms)]

    def _adaptive_lambda(self, context):
        # High-magnitude context -> faster drift -> lower lambda (faster forgetting)
        norm2 = float(np.dot(context, context))
        lam = self.lambda_base + self.gamma * norm2
        if lam > self.lam_max:
            lam = self.lam_max
        if lam < self.lam_min:
            lam = self.lam_min
        return lam

    def select(self, context):
        # Deadline guard: after limit, exploit best-known mean per arm
        if time.time() > self.deadline:
            best = 0
            best_val = -1e18
            for a in range(self.n):
                val = float(self.theta[a] @ context)
                if val > best_val:
                    best_val = val
                    best = a
            return best
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ context)
            # UCB: alpha * sqrt(x^T P x)
            var = float(context @ (self.P[a] @ context))
            if var < 0:
                var = 0.0
            scores.append(mean + self.alpha * np.sqrt(var))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track reward scale per arm for normalization
        self.reward_ema[arm] = self.r_alpha * self.reward_ema[arm] + (1 - self.r_alpha) * reward
        self.reward_scale_ema[arm] = self.r_alpha * self.reward_scale_ema[arm] + (1 - self.r_alpha) * (reward ** 2)
        scale = np.sqrt(self.reward_scale_ema[arm])
        if scale < 1e-6:
            scale = 1.0

        # Innovation (prediction error)
        pred = float(self.theta[arm] @ context)
        innovation = reward - pred
        self.innov_ema[arm] = self.innov_alpha * self.innov_ema[arm] + (1 - self.innov_alpha) * (innovation ** 2)
        # Adaptive process noise: if innovation large, increase forgetting (lower lambda)
        innov_factor = 1.0 + min(0.5, self.innov_ema[arm] / (scale * scale + 1e-9))

        lam = self._adaptive_lambda(context)
        # Apply innovation-driven acceleration
        lam = lam / innov_factor
        if lam < self.lam_min:
            lam = self.lam_min
        if lam > self.lam_max:
            lam = self.lam_max

        # RLS update with forgetting
        P = self.P[arm]
        x = context
        # P_new = (1/lam) * (P - P x / (lam + x^T P x))
        Px = P @ x
        denom = lam + float(x @ Px)
        if denom < 1e-9:
            denom = 1e-9
        gain = Px / denom
        P_new = (1.0 / lam) * (P - np.outer(gain, x))
        # Symmetrize for stability
        P_new = 0.5 * (P_new + P_new.T)
        self.P[arm] = P_new

        # theta update
        self.theta[arm] = self.theta[arm] + gain * innovation
        self.counts[arm] += 1
# EVOLVE-BLOCK-END
