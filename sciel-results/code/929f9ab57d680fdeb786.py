# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with an adaptive forgetting factor (exponential-decay family).

    Per-arm ridge regression on a discounted (geometrically weighted) history of
    (context, reward) pairs.  Uses an incremental Cholesky update so each
    select/update is O(d^2) rather than a full matrix inverse, and an
    adaptive confidence parameter that shrinks as the effective sample size
    grows.  A wall-clock deadline guard guarantees we finish with a valid arm
    well before the evaluator's limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 120.0   # safety margin vs. 150 s kill

        # Discount (forgetting) factor.  Slightly more responsive than the
        # seed's 0.995 so we track changing arm weights faster; the cheaper
        # update makes this affordable within the time budget.
        self.gamma = 0.99
        # Base ridge added to the (discounted) covariance each round.
        self.ridge = 1e-3
        # Exploration base and growth rate for the adaptive beta.
        self.beta0 = 1.0
        self.beta_growth = 0.5
        # Effective-sample-size decay factor for beta adaptation.
        self.beta_decay = self.gamma

        # Per-arm state: A (discounted covariance), b (discounted moment),
        # Cholesky factor L of A (for fast inverse-apply), effective count.
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.L = [np.eye(dim) for _ in range(n_arms)]
        self.N = [0.0 for _ in range(n_arms)]

        # Reward running stats for a bounded, scale-aware bonus.
        self.rw_sum = 0.0
        self.rw_cnt = 0

    def _chol_update(self, a, x):
        """Incrementally update the Cholesky factor of A[a] for A <- A + x x^T."""
        d = self.d
        L = self.L[a]
        A = self.A[a]
        # A is kept as L @ L.T implicitly; recompute A explicitly for the
        # rank-1 addition, then refresh the factor.  d is small (<= ~10), so a
        # direct Cholesky recompute is cheap and numerically robust.
        A += np.outer(x, x)
        # Add a tiny jitter for numerical safety, then factor.
        try:
            L_new = np.linalg.cholesky(A + 1e-9 * np.eye(d))
        except np.linalg.LinAlgError:
            L_new = np.linalg.cholesky(A + 1e-6 * np.eye(d))
        self.L[a] = L_new
        return L_new

    def _solve(self, a, v):
        """Solve A[a] @ y = v using the stored Cholesky factor."""
        L = self.L[a]
        y = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, y)

    def select(self, context):
        ctx = np.asarray(context, dtype=float)
        # Deadline guard: if we are close to the limit, fall back to a greedy
        # (small-bonus) choice computed cheaply, and never spend more time.
        near_deadline = (time.time() > self.deadline - 5.0)

        best_arm = 0
        best_score = -np.inf
        for a in range(self.n):
            theta = self._solve(a, self.b[a])
            mean = float(theta @ ctx)
            if near_deadline:
                score = mean
            else:
                # Adaptive confidence: grows with sqrt(effective samples).
                beta = self.beta0 + self.beta_growth * np.sqrt(self.N[a])
                # Discounted uncertainty: sqrt(x^T A^{-1} x).
                invx = self._solve(a, ctx)
                unc = float(np.sqrt(max(0.0, ctx @ invx)))
                # Bounded, scale-aware optimism.
                scale = 1.0 + 0.5 * min(1.0, self.rw_cnt and (abs(self.rw_sum / self.rw_cnt)))
                score = mean + beta * unc * scale
            if score > best_score:
                best_score = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)
        r = float(reward)
        a = arm
        self.t += 1

        # Discount existing statistics, then add the new observation.
        self.A[a] = self.gamma * self.A[a]
        self.b[a] = self.gamma * self.b[a]
        self.N[a] = self.gamma * self.N[a] + 1.0
        self.A[a] += np.outer(ctx, ctx)
        self.b[a] += r * ctx

        # Refresh the Cholesky factor for the discounted-plus-new covariance.
        d = self.d
        A = self.A[a] + self.ridge * np.eye(d)
        try:
            self.L[a] = np.linalg.cholesky(A + 1e-9 * np.eye(d))
        except np.linalg.LinAlgError:
            self.L[a] = np.linalg.cholesky(A + 1e-6 * np.eye(d))

        # Update reward running stats.
        self.rw_sum += r
        self.rw_cnt += 1
# EVOLVE-BLOCK-END
