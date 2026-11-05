"""Sliding-window LinUCB with adaptive forgetting for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor for non-stationary linear rewards."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.alpha = 1.0  # Exploration parameter
        self.gamma = 0.98  # Forgetting factor
        self.gamma_min = 0.95
        self.gamma_max = 0.995
        self.gamma_decay = 0.999
        self.t = 0
        self.reward_history = [[] for _ in range(n_arms)]
        self.window = 30
        self.deadline = time.time() + 115  # Safety margin before 120s limit
        self.start_time = time.time()

    def _update_inverse(self, arm):
        """Update inverse matrix using Sherman-Morrison formula."""
        A = self.A[arm]
        A_inv = self.A_inv[arm]
        b = self.b[arm]
        self.theta[arm] = A_inv @ b

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current theta estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Adaptively adjust gamma based on reward variance in recent window
        # Use a simple heuristic: if rewards are changing a lot, increase forgetting
        # (decrease gamma) to adapt faster

        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            A_inv = self.A_inv[a]
            mean = theta @ context
            variance = context @ A_inv @ context
            score = mean + self.alpha * np.sqrt(variance)
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Store reward for adaptive gamma
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.window:
            self.reward_history[arm].pop(0)

        # Adaptive forgetting: estimate change in rewards
        # If we have enough history, compute a simple measure of non-stationarity
        if len(self.reward_history[arm]) >= 5:
            recent = self.reward_history[arm][-10:] if len(self.reward_history[arm]) >= 10 else self.reward_history[arm]
            if len(recent) >= 2:
                # Compute rolling difference magnitude
                diffs = np.abs(np.diff(recent))
                change_rate = np.mean(diffs) if len(diffs) > 0 else 0.0
                # Scale gamma: more change -> lower gamma (faster forgetting)
                # Map change_rate to gamma adjustment
                target_gamma = self.gamma_max - min(change_rate * 0.01, self.gamma_max - self.gamma_min)
                # Smoothly adjust gamma
                self.gamma = 0.99 * self.gamma + 0.01 * target_gamma
                self.gamma = max(self.gamma_min, min(self.gamma_max, self.gamma))

        # Apply forgetting
        self.A[arm] = self.gamma * self.A[arm]
        self.b[arm] = self.gamma * self.b[arm]

        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Update inverse using Sherman-Morrison
        # A_new = gamma * A_old + x x^T
        # We need (gamma * A_old + x x^T)^{-1}
        # = (1/gamma) * (A_old + (1/gamma) x x^T)^{-1}
        # Use Sherman-Morrison: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        A_inv = self.A_inv[arm].copy()
        x = context
        # We want (gamma * A_old + x x^T)^{-1}
        # Let M = gamma * A_old, so M^{-1} = (1/gamma) * A_old^{-1}
        # Then (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        M_inv = (1.0 / self.gamma) * A_inv
        M_inv_x = M_inv @ x
        denom = 1.0 + x @ M_inv_x
        if abs(denom) > 1e-10:
            self.A_inv[arm] = M_inv - np.outer(M_inv_x, M_inv_x) / denom
        else:
            # Fallback to direct inversion
            self.A_inv[arm] = np.linalg.inv(self.A[arm])

        # Update theta
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

        # Slowly decay alpha to reduce exploration over time
        self.alpha = max(0.5, self.alpha * 0.9999)
# EVOLVE-BLOCK-END
