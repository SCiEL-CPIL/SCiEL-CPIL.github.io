"""Non-stationary linear contextual bandit with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Discounted LinUCB with adaptive exploration and non-stationarity tracking."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.97  # Forgetting factor
        self.lam = 1.0     # Regularization
        self.alpha_base = 1.0  # Base exploration coefficient
        self.alpha_min = 0.1
        self.alpha_max = 3.0

        # Per-arm state
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]

        # Non-stationarity tracking
        self.residuals = [np.zeros(dim) for _ in range(n_arms)]
        self.residual_sums = [0.0 for _ in range(n_arms)]
        self.residual_sq_sums = [0.0 for _ in range(n_arms)]
        self.last_reward = [None for _ in range(n_arms)]

        # Adaptive alpha
        self.alpha = self.alpha_base
        self.total_rounds = 0
        self.reward_history = []

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin before 120s limit

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Check deadline - if close, use greedy (best estimate)
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    inv_A = np.linalg.inv(self.A[a])
                    theta = inv_A @ self.b[a]
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return int(best_arm)

        scores = []
        for a in range(self.n):
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
                # UCB score
                variance = context @ inv_A @ context
                uncertainty = self.alpha * np.sqrt(max(variance, 0.0))
                score = theta @ context + uncertainty
            except np.linalg.LinAlgError:
                score = 0.0
            scores.append(score)

        # Select arm with highest score
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.total_rounds += 1
        self.reward_history.append(reward)

        # Keep reward history bounded
        if len(self.reward_history) > 200:
            self.reward_history = self.reward_history[-200:]

        # Update per-arm state with discounting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        self.counts[arm] += 1

        # Track residuals for non-stationarity detection
        try:
            inv_A = np.linalg.inv(self.A[arm])
            theta = inv_A @ self.b[arm]
            predicted = theta @ context
            residual = reward - predicted
            self.residuals[arm] = self.gamma * self.residuals[arm] + residual
            self.residual_sums[arm] = self.gamma * self.residual_sums[arm] + abs(residual)
            self.residual_sq_sums[arm] = self.gamma * self.residual_sq_sums[arm] + residual * residual
        except np.linalg.LinAlgError:
            pass

        # Adaptive alpha: reduce exploration as we learn more,
        # but increase if residuals are growing (non-stationarity)
        if self.total_rounds > 10:
            # Estimate residual variance across all arms
            total_res_sq = sum(self.residual_sq_sums[a] for a in range(self.n))
            total_res = sum(self.residual_sums[a] for a in range(self.n))
            total_count = sum(self.counts[a] for a in range(self.n))

            if total_count > 0 and total_res > 0:
                # Normalize by effective sample size
                eff_count = total_count * (1 - self.gamma) / (1 - self.gamma)  # simplified
                mean_res = total_res / max(eff_count, 1)
                var_res = total_res_sq / max(eff_count, 1) - mean_res * mean_res
                var_res = max(var_res, 1e-6)

                # Adaptive alpha: higher when variance is high (non-stationary)
                # Scale by inverse sqrt of effective rounds
                decay = 1.0 / np.sqrt(1 + self.total_rounds * 0.01)
                noise_scale = np.sqrt(var_res)

                # Combine: base exploration decays, but noise adds back
                self.alpha = np.clip(
                    self.alpha_base * decay + 0.5 * noise_scale,
                    self.alpha_min,
                    self.alpha_max
                )
        else:
            # Early rounds: moderate exploration
            self.alpha = self.alpha_base

# EVOLVE-BLOCK-END
