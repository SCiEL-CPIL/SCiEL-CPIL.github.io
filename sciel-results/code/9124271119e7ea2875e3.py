"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Per-Arm Adaptive Discounted Recursive Least Squares (AD-RLS) policy."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 115.0  # safety margin before 120s kill

        # Per-arm state
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) for _ in range(n_arms)]  # covariance (inverse of A)

        # Adaptive forgetting: track innovation per arm
        self.innovation_hist = [deque(maxlen=20) for _ in range(n_arms)]
        self.base_lambda = 0.995
        self.min_lambda = 0.95
        self.max_lambda = 0.999

        # Exploration
        self.base_alpha = 0.5
        self.counts = np.zeros(n_arms)
        self.total_count = 0

        # Reset mechanism
        self.reset_interval = 200
        self.last_reset = 0
        self.round = 0

        # Observation noise estimate
        self.R = 0.1

    def _adaptive_lambda(self, arm):
        """Compute adaptive forgetting factor based on recent innovation variance."""
        if len(self.innovation_hist[arm]) < 5:
            return self.base_lambda

        innovations = np.array(self.innovation_hist[arm])
        var = np.var(innovations)
        mean_abs = np.mean(np.abs(innovations))

        # Higher innovation variance -> lower lambda (faster forgetting)
        # Normalize: if var is high relative to expected, decrease lambda
        scale = max(var, 1e-8)
        # Map variance to lambda: high variance -> lambda near min_lambda
        # Low variance -> lambda near max_lambda
        # Use a smooth mapping
        normalized_var = min(scale / (1.0 + scale), 1.0)
        lambda_val = self.max_lambda - normalized_var * (self.max_lambda - self.min_lambda)
        return max(self.min_lambda, min(self.max_lambda, lambda_val))

    def _compute_theta(self, arm):
        """Compute theta from P and b."""
        return self.P[arm] @ self.b[arm]

    def select(self, context):
        self.round += 1

        # Deadline guard
        if time.time() > self.deadline:
            # Return best arm based on current theta
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        # Periodic reset for non-stationarity
        if self.round - self.last_reset >= self.reset_interval and self.total_count > 50:
            for a in range(self.n):
                self.A[a] = 0.5 * self.A[a] + 0.5 * np.eye(self.d)
                self.b[a] = 0.5 * self.b[a]
                self.P[a] = np.linalg.inv(self.A[a])
                self.theta[a] = self.P[a] @ self.b[a]
                self.innovation_hist[a].clear()
            self.last_reset = self.round

        scores = []
        ctx = np.asarray(context, dtype=np.float64).ravel()

        for a in range(self.n):
            # Count-scaled UCB with time decay
            count = max(self.counts[a], 1)
            alpha = self.base_alpha * np.sqrt(np.log(1 + self.total_count) / count)
            # Decay exploration over time
            alpha *= np.exp(-0.001 * self.round)

            theta = self.theta[a]
            mean_reward = theta @ ctx

            # Uncertainty term
            var = ctx @ self.P[a] @ ctx
            std = np.sqrt(max(var, 0.0))

            scores.append(mean_reward + alpha * std)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        self.counts[arm] += 1
        self.total_count += 1

        ctx = np.asarray(context, dtype=np.float64).ravel()

        # Compute innovation
        theta = self.theta[arm]
        predicted = theta @ ctx
        innovation = reward - predicted

        # Track innovation for adaptive lambda
        self.innovation_hist[arm].append(innovation)

        # Get adaptive forgetting factor
        lam = self._adaptive_lambda(arm)

        # Recursive least squares update with forgetting
        # P_new = (1/lam) * (P - P @ x @ (lam + x^T @ P @ x)^-1 @ x^T @ P)
        # Using matrix inversion lemma for efficiency

        P = self.P[arm]
        A = self.A[arm]
        b = self.b[arm]

        # Update A and b with forgetting
        A_new = lam * A + np.outer(ctx, ctx)
        b_new = lam * b + reward * ctx

        # Update P = inv(A_new)
        # Use Sherman-Morrison for rank-1 update if possible, but for stability:
        # P_new = inv(lam * A + xx^T)
        # = inv(lam*A) * (1 - xx^T * (lam*A + xx^T)^-1) ... complex
        # Simpler: use the RLS formula directly
        # P_new = (1/lam) * P - (1/lam) * P @ x @ (1 + x^T @ P @ x / lam)^-1 @ x^T @ P / lam

        # Let's use the standard RLS update:
        # K = P @ x / (lam + x^T @ P @ x)
        # P_new = (P - K @ x^T @ P) / lam
        # theta_new = theta + K * innovation

        Px = P @ ctx
        denom = lam + ctx @ Px
        K = Px / denom

        P_new = (P - np.outer(K, Px)) / lam
        theta_new = theta + K * innovation

        # Also update A and b for consistency (in case we need them)
        A_new = lam * A + np.outer(ctx, ctx)
        b_new = lam * b + reward * ctx

        self.P[arm] = P_new
        self.theta[arm] = theta_new
        self.A[arm] = A_new
        self.b[arm] = b_new
# EVOLVE-BLOCK-END
