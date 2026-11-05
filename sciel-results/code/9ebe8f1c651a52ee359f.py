"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Kernel Recursive Least Squares (KRLS).
    Uses RBF kernels with a fixed dictionary of centers.
    Incorporates adaptive forgetting factor to handle regime shifts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.M = 5  # Number of RBF centers per arm
        self.gamma = 1.0  # RBF kernel width
        
        # Per-arm state
        self.centers = [np.zeros((self.M, self.d)) for _ in range(n_arms)]
        self.w = [np.zeros(self.M + 1) for _ in range(n_arms)]  # +1 for bias
        self.P = [np.eye(self.M + 1) * 10.0 for _ in range(n_arms)] # Initial inverse covariance
        
        # Hyperparameters
        self.lam_base = 0.95
        self.lam = self.lam_base
        self.R = 1e-2
        self.beta = 0.01  # Adaptation rate for lam
        
        # Tracking
        self.step = 0
        self.deadline = time.time() + 110.0 # Safety margin
        
        # Initialize centers to spread out slightly
        for a in range(n_arms):
            self.centers[a] = np.random.randn(self.M, self.d) * 0.1

    def _phi(self, x, centers):
        """Compute RBF feature vector."""
        # x is (d,), centers is (M, d)
        dists = np.sum((x[None, :] - centers) ** 2, axis=1)
        k_vals = np.exp(-self.gamma * dists)
        return np.concatenate(([1.0], k_vals))

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback

        scores = []
        for a in range(self.n):
            phi = self._phi(context, self.centers[a])
            mu = self.w[a] @ phi
            var = phi @ self.P[a] @ phi
            # UCB with adaptive exploration
            # Use a constant exploration factor for stability
            scores.append(mu + 1.5 * np.sqrt(max(var, 1e-8)))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() > self.deadline:
            return

        self.step += 1
        phi = self._phi(context, self.centers[arm])
        
        # Prediction
        pred = self.w[arm] @ phi
        error = reward - pred

        # Adaptive Forgetting Factor
        # If error is large, reduce lambda (forget faster)
        # Simple heuristic: lambda_t = lambda_base - beta * |error|
        # Clip to [0.5, 0.99]
        self.lam = self.lam_base - self.beta * abs(error)
        self.lam = np.clip(self.lam, 0.5, 0.99)

        # RLS Update
        Pphi = self.P[arm] @ phi
        denom = self.R + phi @ Pphi
        if denom < 1e-8:
            denom = 1e-8
        K = Pphi / denom
        
        self.w[arm] = self.w[arm] + K * error
        self.P[arm] = (self.P[arm] - np.outer(K, phi)) / self.lam
        
        # Add small regularization to prevent P from becoming singular or too small
        self.P[arm] += 1e-6 * np.eye(self.P[arm].shape[0])

        # Update Centers Periodically
        # Every 20 steps, update one center to the current context
        if self.step % 20 == 0:
            idx = (self.step // 20) % self.M
            self.centers[arm][idx] = context

# EVOLVE-BLOCK-END
