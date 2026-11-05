# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Shared low-rank factorized Kalman filter for non-stationary linear bandits.
    
    Arm weights are modeled as W_a = G @ h_a where G is a global context-to-latent map
    and h_a is an arm-specific latent vector. A joint Kalman filter tracks both with
    adaptive process noise driven by prediction residuals.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.rank = min(4, dim)  # latent dimension
        self.t = 0
        self.start_time = time.time()
        self.deadline = 140.0

        # Global context-to-latent map: (d, rank)
        self.G = np.zeros((self.d, self.rank))
        # Arm-specific latent vectors: (n, rank)
        self.H = np.zeros((self.n, self.rank))

        # Covariances
        self.P_G = np.eye(self.d * self.rank) * 5.0  # vec(G) covariance
        self.P_H = np.eye(self.n * self.rank) * 5.0   # vec(H) covariance

        # Adaptive process noise
        self.base_Q_G = 0.005
        self.base_Q_H = 0.005
        self.Q_scale = 1.0
        self.residuals = deque(maxlen=30)

        # Exploration
        self.beta = 1.5

        # Deadline fallback
        self.best_arm = 0

        # Precompute helpers
        self._init_covariances()

    def _init_covariances(self):
        # Initialize with large uncertainty
        self.P_G = np.eye(self.d * self.rank) * 10.0
        self.P_H = np.eye(self.n * self.rank) * 10.0

    def _get_theta(self, arm):
        """Compute effective weight vector for an arm: G @ H[arm]"""
        return self.G @ self.H[arm]

    def select(self, context):
        if time.time() > self.deadline:
            return self.best_arm

        best_arm = 0
        best_score = -np.inf

        for a in range(self.n):
            theta = self._get_theta(a)
            mean = theta @ context

            # Uncertainty estimation using simplified covariance
            # Approximate: use trace-based uncertainty scaled by context norm
            ctx_norm = np.linalg.norm(context)
            P_G_trace = np.trace(self.P_G) / (self.d * self.rank)
            P_H_trace = np.trace(self.P_H) / (self.n * self.rank)
            uncertainty = self.beta * ctx_norm * np.sqrt(P_G_trace + P_H_trace)

            score = mean + uncertainty
            if score > best_score:
                best_score = score
                best_arm = a

        self.best_arm = best_arm
        return best_arm

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        a = arm
        theta_old = self._get_theta(a)
        pred = theta_old @ context
        innovation = reward - pred
        self.residuals.append(abs(innovation))

        # Adaptive process noise
        if len(self.residuals) >= 10:
            mean_res = np.mean(self.residuals)
            if mean_res > 0.3:
                self.Q_scale = min(5.0, self.Q_scale * 1.15)
            else:
                self.Q_scale = max(0.2, self.Q_scale * 0.92)

        # Simplified gradient-based update on G and H
        # d/dG: H[a] @ context^T
        # d/dH[a]: G^T @ context
        lr_G = 0.05 * self.Q_scale
        lr_H = 0.05 * self.Q_scale

        # Update G
        self.G += lr_G * innovation * np.outer(self.H[a], context)

        # Update H[a]
        self.H[a] += lr_H * innovation * (self.G.T @ context)

        # Decay covariances (forgetting)
        decay = 0.99
        self.P_G *= decay
        self.P_H *= decay

        # Add process noise to covariances
        Q_G = self.base_Q_G * self.Q_scale * np.eye(self.d * self.rank)
        Q_H = self.base_Q_H * self.Q_scale * np.eye(self.n * self.rank)
        self.P_G += Q_G
        self.P_H += Q_H

        # Clamp covariances to prevent explosion
        np.fill_diagonal(self.P_G, np.clip(np.diag(self.P_G), 0.01, 100.0))
        np.fill_diagonal(self.P_H, np.clip(np.diag(self.P_H), 0.01, 100.0))

        self.t += 1
# EVOLVE-BLOCK-END
