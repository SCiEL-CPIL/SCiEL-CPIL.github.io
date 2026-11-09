"""Non-stationary linear contextual bandit policy.

Design (from scratch):
- Exponential-discounting Recursive Least Squares (RLS) per arm.
- We maintain the *inverse* covariance V = A^{-1} directly and update it with a
  Sherman-Morrison rank-1 update, giving O(d^2) per update (no O(d^3) inversions).
- UCB exploration bonus for stable exploitation in the noisy regime.
- An explicit wall-clock deadline guard returns the best-so-far arm before the
  120 s limit, with a safety margin.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Discounted RLS linear UCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Discount (forgetting) factor: < 1 forgets old data exponentially.
        # A value near 1 keeps long memory but still adapts; tune for non-stationarity.
        self.gamma = 0.99
        # Regularization to keep V positive definite and well-conditioned.
        self.reg = 1.0
        # UCB exploration coefficient.
        self.beta = 1.0

        # V[a] = A[a]^{-1}, b[a] = A[a]^{-1} * (reward-weighted context sum).
        # We store V and b so that theta = V @ b is the RLS weight estimate.
        self.V = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Deadline guard: track start time; stop heavy work before the limit.
        self._t0 = time.time()
        self._deadline = self._t0 + 110.0  # 10 s safety margin before 120 s
        self._past_deadline = False

    def _expired(self):
        return self._past_deadline or (time.time() >= self._deadline)

    def select(self, context):
        ctx = np.asarray(context, dtype=float).ravel()

        # Deadline guard: if we are about to run out of time, pick greedily
        # using the current weight estimates (cheap) and stop adapting.
        if self._expired():
            best_a = 0
            best_val = -np.inf
            for a in range(self.n):
                val = (self.V[a] @ self.b[a]) @ ctx
                if val > best_val:
                    best_val = val
                    best_a = a
            self._past_deadline = True
            return int(best_a)

        scores = np.empty(self.n)
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ ctx
            # UCB bonus from the posterior variance of the prediction.
            var = self.reg * ctx @ self.V[a] @ ctx
            scores[a] = mu + self.beta * np.sqrt(max(var, 0.0))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Once past the deadline, stop learning to save time.
        if self._expired():
            self._past_deadline = True
            return

        ctx = np.asarray(context, dtype=float).ravel()
        g = self.gamma
        r = float(reward)

        # Exponential discount of existing statistics.
        self.V[arm] = g * self.V[arm]
        self.b[arm] = g * self.b[arm]

        # Sherman-Morrison rank-1 update for V = (g*V)^{-1} form.
        # We maintain V as the inverse of the (discounted) covariance.
        # Update: V_new = V - (V x)(V x)^T / (lambda + x^T V x)
        # where we incorporate the new observation with regularization lambda.
        V = self.V[arm]
        vx = V @ ctx
        denom = self.reg + ctx @ vx
        if denom <= 1e-12:
            denom = 1e-12
        V = V - np.outer(vx, vx) / denom
        self.V[arm] = V

        # Update b: b_new = V_new @ (b + r * x)  (RLS mean update)
        self.b[arm] = V @ (self.b[arm] + r * ctx)
# EVOLVE-BLOCK-END
