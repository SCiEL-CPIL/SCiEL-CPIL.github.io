"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm RLS with context-adaptive forgetting factor and count-scaled UCB."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145  # 145s deadline guard

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Hyperparameters
        self.lam_base = 0.98
        self.gamma = 0.001  # context-adaptive scaling
        self.alpha = 1.0    # UCB exploration coefficient
        self.beta = 0.1     # noise variance estimate
        self.min_count = 10  # minimum counts before full exploration

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return the arm with highest mean reward estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                if self.counts[a] > 0:
                    score = self.theta[a] @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
            return best_arm

        self.t += 1
        x = np.asarray(context, dtype=np.float64)
        x_norm_sq = float(x @ x)

        # Context-adaptive forgetting factor
        lam = self.lam_base + self.gamma * x_norm_sq
        lam = min(lam, 0.9999)  # cap to avoid instability

        scores = []
        for a in range(self.n):
            # Estimate
            mu = self.theta[a] @ x

            # Uncertainty: sqrt(x^T P x)
            Px = self.P[a] @ x
            variance = float(x @ Px)
            std = np.sqrt(max(variance, 0.0))

            # Count-scaled UCB with time decay
            count = self.counts[a]
            if count < self.min_count:
                # Early exploration: larger bonus
                bonus = self.alpha * std * np.sqrt(self.min_count / max(count, 1))
            else:
                # Later: decay exploration with time
                bonus = self.alpha * std / np.sqrt(self.t)

            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64)
        self.counts[arm] += 1

        # Context-adaptive forgetting factor
        x_norm_sq = float(x @ x)
        lam = self.lam_base + self.gamma * x_norm_sq
        lam = min(lam, 0.9999)

        # RLS update with forgetting
        # P = (1/lam) * P - (1/lam) * P x (x^T P + lam * I)^{-1} x^T P / lam
        # Simplified: P_new = (1/lam) P - (1/lam^2) P x x^T P / (x^T P x / lam + 1)

        P = self.P[arm]
        theta = self.theta[arm]

        # Compute x^T P x
        Px = P @ x
        xPx = float(x @ Px)

        # Effective denominator
        denom = xPx / lam + 1.0

        # Update P
        # P_new = (1/lam) * P - (1/lam^2) * P x x^T P / denom
        # P_new = (1/lam) * P - (1/lam^2) * (P x) (x^T P) / denom
        # Note: P is symmetric, so x^T P = (P x)^T
        outer = np.outer(Px, Px)  # P x x^T P
        P_new = (1.0 / lam) * P - (1.0 / (lam * lam)) * outer / denom

        # Update theta
        # theta_new = (1/lam) * theta + P_new * (reward - x^T theta) * x / denom
        # Actually: theta_new = theta + P_new * (reward - x^T theta_old) * x
        # But with forgetting, it's: theta_new = (1/lam)*theta + P_new*(r - x^T theta)*x
        # Let me use the standard RLS with forgetting:
        # K = P x / (lam + x^T P x)
        # theta_new = theta + K * (r - x^T theta)
        # P_new = (1/lam)(P - K x^T P)

        # Let me redo with the standard formulation:
        # K = P x / (lam + x^T P x)
        K = Px / (lam + xPx)

        # Prediction error
        pred = float(x @ theta)
        error = reward - pred

        # Update theta
        theta_new = theta + K * error

        # Update P
        P_new = (1.0 / lam) * (P - np.outer(K, x) @ P)
        # Ensure symmetry
        P_new = 0.5 * (P_new + P_new.T)

        self.P[arm] = P_new
        self.theta[arm] = theta_new
# EVOLVE-BLOCK-END
