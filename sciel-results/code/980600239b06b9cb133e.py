"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Recurrent Linear Bandit: augments context with previous reward and arm to capture non-stationarity."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: original context + 1 (prev reward) + n_arms (one-hot prev arm)
        self.aug_dim = dim + 1 + n_arms
        self.A = [np.eye(self.aug_dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(self.aug_dim) for _ in range(n_arms)]
        self.L = [np.linalg.cholesky(self.A[i]) for i in range(n_arms)]
        self.L_inv = [np.linalg.inv(self.L[i]) for i in range(n_arms)]
        self.alpha = 0.95
        self.sigma_sq = 1.0
        self._start_time = time.time()
        self._deadline = 145.0  # safety margin before 150s limit
        self._round_count = 0
        self._prev_reward = 0.0
        self._prev_arm = None

    def _augment_context(self, context):
        """Augment context with previous reward and one-hot previous arm."""
        aug = np.zeros(self.aug_dim)
        aug[:self.d] = context
        aug[self.d] = self._prev_reward
        if self._prev_arm is not None:
            aug[self.d + 1 + self._prev_arm] = 1.0
        else:
            # Initialize with uniform distribution over arms
            aug[self.d + 1:self.aug_dim] = 1.0 / self.n
        return aug

    def _compute_posterior(self, arm, context):
        """Compute posterior mean and sample from it using Cholesky."""
        A = self.A[arm]
        b = self.b[arm]
        L = self.L[arm]
        L_inv = self.L_inv[arm]
        try:
            # theta_hat = L^{-T} * L^{-1} * b
            y = L_inv @ b
            theta_hat = L_inv.T @ y
            # Sample: theta = theta_hat + sigma * L^{-T} * z where z ~ N(0, I)
            z = np.random.randn(self.aug_dim)
            L_inv_T_z = L_inv.T @ z
            theta = theta_hat + self.sigma_sq * L_inv_T_z
            return theta
        except np.linalg.LinAlgError:
            # Fallback to direct inverse if Cholesky fails
            try:
                inv_A = np.linalg.inv(A)
                theta_hat = inv_A @ b
                z = np.random.randn(self.aug_dim)
                cov = self.sigma_sq * inv_A
                L_cov = np.linalg.cholesky(cov)
                theta = theta_hat + L_cov @ z
                return theta
            except np.linalg.LinAlgError:
                try:
                    inv_A = np.linalg.pinv(A)
                    return inv_A @ b
                except:
                    return np.zeros(self.aug_dim)

    def select(self, context):
        self._round_count += 1
        # Deadline guard: if we're close to the limit, exploit greedily
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline:
            # Greedy fallback: use posterior means without sampling
            aug_ctx = self._augment_context(context)
            scores = []
            for a in range(self.n):
                try:
                    inv_A = np.linalg.inv(self.A[a])
                    theta = inv_A @ self.b[a]
                    scores.append(theta @ aug_ctx)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        # Thompson Sampling: sample from posterior for each arm
        aug_ctx = self._augment_context(context)
        scores = []
        for a in range(self.n):
            theta = self._compute_posterior(a, context)
            scores.append(theta @ aug_ctx)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update with forgetting factor
        aug_ctx = self._augment_context(context)
        self.A[arm] = self.alpha * self.A[arm] + np.outer(aug_ctx, aug_ctx)
        self.b[arm] = self.alpha * self.b[arm] + reward * aug_ctx
        
        # Update Cholesky factorization
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            self.L_inv[arm] = np.linalg.inv(self.L[arm])
        except np.linalg.LinAlgError:
            pass  # Keep old factorization if update fails
        
        # Update state for next round
        self._prev_reward = reward
        self._prev_arm = arm
# EVOLVE-BLOCK-END
