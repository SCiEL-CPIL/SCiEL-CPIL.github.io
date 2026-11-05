# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Per-arm Kalman filter (random-walk weights) with adaptive process noise.

    Each arm keeps a mean weight vector theta and a covariance P. Between
    observations the state performs a random-walk "predict" step adding
    process noise Q; an observation performs a rank-1 "update". Q adapts to
    how surprised the model is by the latest reward (innovation), so it reacts
    quickly to non-stationary weight changes but stays small when stable.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty so unexplored arms are favoured.
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Tunable constants (tuned for the 5-arm, low-dim setting).
        self.q0 = 0.02          # base process noise
        self.q_max = 3.0        # cap on adaptive process noise
        self.innov_gain = 0.05  # how strongly surprise raises Q
        self.u_scale = 1.0      # UCB exploration multiplier
        self.count_scale = 0.4  # sqrt(count) dampening of the bonus
        self.jitter = 1e-6      # numerical stabiliser for P
        self.deadline = time.time() + 110.0  # safety margin before the 120s kill

    def _predict(self, a, q):
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter

    def _update(self, a, ctx, reward):
        P = self.P[a]
        theta = self.theta[a]
        pred = float(theta @ ctx)
        innov = reward - pred
        var = float(ctx @ P @ ctx) + self.jitter
        K = P @ ctx / var          # Kalman gain
        self.theta[a] = theta + (K * innov)
        # Joseph-form rank-1 update: P <- (I - K x') P
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T
        # Symmetrise and clip for numerical safety.
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 0.0))
        self.count[a] += 1
        return innov

    def select(self, context):
        ctx = np.asarray(context, dtype=float)
        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest current mean estimate (no further exploration).
        if time.time() > self.deadline:
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            std = float(np.sqrt(max(ctx @ self.P[a] @ ctx, 0.0)))
            c = self.count[a]
            bonus = self.u_scale * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 1.0  # encourage exploring unseen arms early
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)
        self.t += 1
        # Adapt process noise from the innovation before predicting.
        innov = reward - float(self.theta[arm] @ ctx)
        q = min(self.q_max, self.q0 + self.innov_gain * abs(innov))
        self._predict(arm, q)
        self._update(arm, ctx, reward)
# EVOLVE-BLOCK-END
