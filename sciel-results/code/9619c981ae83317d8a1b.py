"""Non-stationary linear contextual bandit policy combining hybrid UCB/Thompson Sampling
with recurrent context augmentation and adaptive drift detection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    1. Recurrent context augmentation (append previous reward and arm one-hot)
    2. Hybrid UCB + Thompson Sampling exploration
    3. Adaptive forgetting factor based on residual variance
    4. Drift detection with model reset
    5. Cholesky decomposition for numerical stability
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: original dim + 1 (prev reward) + n_arms (arm one-hot)
        self.d_aug = dim + 1 + n_arms
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Hyperparameters
        self.gamma_base = 0.995
        self.gamma_min = 0.97
        self.gamma_decay = 500
        self.beta = 1.2
        self.ts_weight = 0.6
        self.reset_threshold = 2.0
        self.ewma_alpha = 0.15
        self.gamma = self.gamma_base

        # Initialize per-arm models with augmented dimension
        eye = np.eye(self.d_aug)
        self.A = [eye.copy() for _ in range(n_arms)]
        self.b = [np.zeros(self.d_aug) for _ in range(n_arms)]
        self.L = [np.linalg.cholesky(eye + 1e-6 * np.eye(self.d_aug)) for _ in range(n_arms)]
        self.L_valid = [True] * n_arms

        # Recurrent state: previous reward and arm
        self.prev_reward = 0.0
        self.prev_arm = 0

        # Residual tracking for drift detection
        self.residuals = [np.zeros(1) for _ in range(n_arms)]
        self.reward_ewma = [0.0] * n_arms
        self.reward_var_ewma = [0.0] * n_arms

        # RNG for Thompson Sampling
        self._rng = np.random.RandomState(42)

    def _augment_context(self, context):
        """
        Augment context with previous reward and arm one-hot encoding.
        This allows the linear model to capture temporal dependencies.
        """
        context = np.asarray(context, dtype=np.float64).flatten()
        # Create one-hot for previous arm
        arm_onehot = np.zeros(self.n)
        arm_onehot[self.prev_arm] = 1.0
        # Augment: [context, prev_reward, arm_onehot]
        return np.concatenate([context, [self.prev_reward], arm_onehot])

    def _update_cholesky(self, arm):
        """Recompute Cholesky factor for arm's A matrix."""
        A = self.A[arm]
        M = A + 1e-6 * np.eye(self.d_aug)
        try:
            self.L[arm] = np.linalg.cholesky(M)
            self.L_valid[arm] = True
        except np.linalg.LinAlgError:
            # Try larger regularization
            M = A + 1e-3 * np.eye(self.d_aug)
            try:
                self.L[arm] = np.linalg.cholesky(M)
                self.L_valid[arm] = True
            except np.linalg.LinAlgError:
                self.L_valid[arm] = False

    def _solve_cholesky(self, arm, rhs):
        """Solve L L^T x = rhs using Cholesky factor."""
        if self.L_valid[arm]:
            y = np.linalg.solve(self.L[arm], rhs)
            x = np.linalg.solve(self.L[arm].T, y)
            return x
        else:
            return np.linalg.solve(self.A[arm] + 1e-6 * np.eye(self.d_aug), rhs)

    def _variance_cholesky(self, arm, context):
        """Compute context^T A^{-1} context using Cholesky factor."""
        if self.L_valid[arm]:
            z = np.linalg.solve(self.L[arm].T, context)
            var = float(np.dot(z, z))
            return max(var, 0.0)
        else:
            inv_A = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.d_aug))
            var = float(context @ inv_A @ context)
            return max(var, 0.0)

    def _adapt_forgetting(self, arm, reward, pred):
        """Adapt forgetting factor based on non-stationarity detection."""
        # Update EWMA of reward
        if self.t > 0:
            self.reward_ewma[arm] = (1 - self.ewma_alpha) * self.reward_ewma[arm] + self.ewma_alpha * reward
            diff = reward - self.reward_ewma[arm]
            self.reward_var_ewma[arm] = (1 - self.ewma_alpha) * self.reward_var_ewma[arm] + self.ewma_alpha * (diff ** 2)
        else:
            self.reward_ewma[arm] = reward
            self.reward_var_ewma[arm] = 0.0

        # Residual-based non-stationarity measure
        resid = reward - pred
        resid_sq = resid * resid

        # Combine variance and residual for gamma adjustment
        if self.reward_var_ewma[arm] > 1e-8:
            norm_var = min(self.reward_var_ewma[arm] / (self.reward_var_ewma[arm] + 1.0), 1.0)
            base_gamma = self.gamma_base - 0.025 * norm_var
        else:
            base_gamma = self.gamma_base

        # Also reduce gamma if residual is large
        if resid_sq > 1.0:
            base_gamma = min(base_gamma, 0.98)

        # Time-based decay
        progress = min(1.0, self.t / self.gamma_decay)
        g = base_gamma - (base_gamma - self.gamma_min) * progress * 0.5

        self.gamma = max(self.gamma_min, min(self.gamma_base, g))

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return the arm with highest estimated mean reward
            aug_ctx = self._augment_context(context)
            scores = []
            for a in range(self.n):
                try:
                    theta = self._solve_cholesky(a, self.b[a])
                    scores.append(float(theta @ aug_ctx))
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        aug_ctx = self._augment_context(context)
        best_arm = 0
        best_score = -np.inf

        for a in range(self.n):
            try:
                # Get theta = A^{-1} b
                theta = self._solve_cholesky(a, self.b[a])
                mean = float(theta @ aug_ctx)

                # Get variance for UCB
                var = self._variance_cholesky(a, aug_ctx)
                std = np.sqrt(var)

                # Hybrid score: UCB + Thompson noise
                noise = self._rng.randn() * self.ts_weight * std
                score = mean + self.beta * std + noise

                if score > best_score:
                    best_score = score
                    best_arm = a
            except:
                continue

        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)
        arm = int(arm)

        # Augment context
        aug_ctx = self._augment_context(context)

        # Compute prediction for residual tracking
        try:
            theta = self._solve_cholesky(arm, self.b[arm])
            pred = float(theta @ aug_ctx)
        except:
            pred = 0.0

        # Adapt forgetting factor
        self._adapt_forgetting(arm, reward, pred)

        # Apply forgetting and update
        gamma = self.gamma
        self.A[arm] = gamma * self.A[arm] + np.outer(aug_ctx, aug_ctx)
        self.b[arm] = gamma * self.b[arm] + reward * aug_ctx

        # Update Cholesky factor
        self._update_cholesky(arm)

        # Drift detection: Update residual EWMA and check for reset
        if self.t > 10:
            resid = reward - pred
            resid_sq = resid * resid

            prev_res = self.residuals[arm][0]
            new_res = (1 - self.ewma_alpha) * prev_res + self.ewma_alpha * resid_sq
            self.residuals[arm] = np.array([new_res])

            # If residual variance is high, reset the model for this arm
            if new_res > self.reset_threshold:
                eye = np.eye(self.d_aug)
                self.A[arm] = eye.copy()
                self.b[arm] = np.zeros(self.d_aug)
                self.L[arm] = np.linalg.cholesky(eye + 1e-6 * np.eye(self.d_aug))
                self.L_valid[arm] = True
                self.residuals[arm] = np.zeros(1)

        # Update recurrent state
        self.prev_reward = reward
        self.prev_arm = arm

        self.t += 1

# EVOLVE-BLOCK-END
