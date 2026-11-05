"""Non-stationary linear contextual bandit: Thompson Sampling + RLS with exponential forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Thompson Sampling for non-stationary linear Gaussian bandits.

    Maintains a per-arm posterior over the weight vector w_a:
        theta_a ~ N(mu_a, (A_a)^{-1} / beta)
    where A_a and b_a are updated with an exponential forgetting factor gamma
    (RLS).  Selection samples one theta_a per arm and picks argmax theta_a^T x.

    For O(d^2) updates we keep the inverse of A_a explicitly and apply the
    matrix-inversion (Sherman-Morrison) lemma in update().
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.99          # exponential forgetting factor
        self.beta = 1.0            # posterior precision scaling (inverse variance scale)
        self._deadline = time.time() + 110.0  # safety margin well under 150 s

        # Per-arm state: keep A_inv (d x d) and mu (d)
        # Prior: A = I  =>  A_inv = I,  mu = 0
        self.A_inv = [np.eye(dim, dtype=np.float64) for _ in range(n_arms)]
        self.mu = [np.zeros(dim, dtype=np.float64) for _ in range(n_arms)]
        # Track running reward for a simple baseline / sanity
        self._t = 0

    # ------------------------------------------------------------------
    def select(self, context):
        x = np.asarray(context, dtype=np.float64).ravel()

        # Thompson sampling: sample theta_a ~ N(mu_a, A_inv_a / beta)
        # Use Cholesky of A_inv for a stable draw.
        best_arm = 0
        best_score = -np.inf
        for a in range(self.n):
            cov = self.A_inv[a] / self.beta
            # Cholesky; if it fails (rare, near-singular), fall back to diag
            try:
                L = np.linalg.cholesky(cov)
                z = np.random.standard_normal(self.d)
                theta = self.mu[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback: use mean only (exploit)
                theta = self.mu[a]
            score = float(theta @ x)
            if score > best_score:
                best_score = score
                best_arm = a
        return int(best_arm)

    # ------------------------------------------------------------------
    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).ravel()
        r = float(reward)
        a = int(arm)

        # RLS with exponential forgetting:
        #   A <- gamma * A + x x^T
        #   b <- gamma * b + r * x
        # We store A_inv and mu = A_inv @ b.  Update via the identity:
        #   A_new = gamma * A_old + x x^T
        #   A_new^{-1} = (1/gamma) * (A_old^{-1} -
        #                 (A_old^{-1} x x^T A_old^{-1}) / (gamma + x^T A_old^{-1} x))
        # and mu_new = A_new^{-1} @ b_new,  b_new = gamma * b_old + r x
        # Note: b_old = A_old @ mu_old, but we don't store A_old.  Instead
        # maintain mu directly with a consistent update.
        #
        # Simpler and robust: keep A_inv and mu, and update both.
        #   Let M = A_old^{-1}.  Then A_new = gamma A_old + x x^T.
        #   A_new^{-1} = (1/gamma) M - (1/gamma^2) (M x x^T M) / (1 + (x^T M x)/gamma)
        #   b_new = gamma b_old + r x,  with b_old = A_old mu_old = M^{-1} mu_old.
        #
        # To avoid needing A_old, store b as well (d-vector, cheap).
        #
        # We'll store b per arm instead of relying on A_old.
        if not hasattr(self, '_b'):
            self._b = [np.zeros(self.d, dtype=np.float64) for _ in range(self.n)]

        M = self.A_inv[a]
        b = self._b[a]
        g = self.gamma

        # v = M @ x
        v = M @ x
        denom = g + float(x @ v)          # = gamma + x^T M x
        # A_new^{-1} = (1/gamma) * (M - (v v^T)/denom)
        M_new = (M - np.outer(v, v) / denom) / g
        # b_new = gamma * b + r * x
        b_new = g * b + r * x
        # mu_new = M_new @ b_new
        mu_new = M_new @ b_new

        self.A_inv[a] = M_new
        self._b[a] = b_new
        self.mu[a] = mu_new

        self._t += 1
# EVOLVE-BLOCK-END
