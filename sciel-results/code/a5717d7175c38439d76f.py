"""Multi-scale ensemble linear bandit with model selection via residual variance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-scale ensemble of linear models for non-stationary linear contextual bandits.
    
    Each arm maintains a bank of regressors with geometrically spaced forgetting factors.
    The policy selects the arm that maximizes the reward prediction from the ensemble member
    with the lowest recent residual variance, effectively using model selection to dynamically
    switch memory horizons.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Geometrically spaced forgetting factors: short memory to long memory
        # 0.7 (fast adaptation), 0.85, 0.95, 0.99, 0.999 (slow adaptation)
        self.forgetting_factors = [0.7, 0.85, 0.95, 0.99, 0.999]
        self.n_scales = len(self.forgetting_factors)

        # For each arm, maintain a bank of (A, b) pairs
        # A[a][s] is the precision matrix for arm a at scale s
        # b[a][s] is the cross-product vector for arm a at scale s
        self.A = [[[np.eye(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]]
        self.b = [[[np.zeros(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]]

        # Track recent residuals for each arm and scale to compute residual variance
        # residual_var[a][s] = exponential moving average of squared residuals
        self.residual_var = [[0.0] * self.n_scales for _ in range(n_arms)]
        # Count of updates per arm-scale for initialization
        self.count = [[0] * self.n_scales for _ in range(n_arms)]

        # Exploration parameter
        self.beta = 1.0

        # Cache for inverse matrices to avoid recomputation
        self.inv_cache = [[[None] * self.n_scales for _ in range(n_arms)]]

    def _get_inverse(self, arm, scale):
        """Get cached inverse of A, recomputing if needed."""
        if self.inv_cache[arm][scale] is None:
            self.inv_cache[arm][scale] = np.linalg.inv(self.A[arm][scale])
        return self.inv_cache[arm][scale]

    def select(self, context):
        """Select arm using the ensemble member with lowest residual variance per arm."""
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if time.time() > self.deadline:
            # Return a simple greedy fallback
            scores = []
            for a in range(self.n):
                inv = self._get_inverse(a, 0)
                theta = inv @ self.b[a][0]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # For each arm, find the scale with the lowest residual variance
        # and use that model's prediction + exploration bonus
        arm_predictions = []
        for a in range(self.n):
            # Find best scale (lowest residual variance)
            # For early rounds, prefer longer memory scales
            if self.count[a][0] < 5:
                best_scale = self.n_scales - 1  # longest memory
            else:
                # Weight residual variance by scale to prefer longer memory when close
                best_scale = int(np.argmin(self.residual_var[a]))

            inv = self._get_inverse(a, best_scale)
            theta = inv @ self.b[a][best_scale]
            mu = theta @ context
            # Exploration bonus based on uncertainty
            uncertainty = np.sqrt(context @ inv @ context)
            arm_predictions.append(mu + self.beta * uncertainty)

        return int(np.argmax(arm_predictions))

    def update(self, context, arm, reward):
        """Update all scales for the given arm."""
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        for s, gamma in enumerate(self.forgetting_factors):
            # Update precision matrix and cross-product
            self.A[arm][s] = gamma * self.A[arm][s] + np.outer(context, context)
            self.b[arm][s] = gamma * self.b[arm][s] + reward * context

            # Invalidate inverse cache
            self.inv_cache[arm][s] = None

            # Update residual variance
            self.count[arm][s] += 1
            if self.count[arm][s] > 1:
                # Compute predicted reward using this scale's model
                inv = np.linalg.inv(self.A[arm][s])
                theta = inv @ self.b[arm][s]
                pred = theta @ context
                residual = reward - pred
                # Exponential moving average of squared residual
                self.residual_var[arm][s] = 0.95 * self.residual_var[arm][s] + 0.05 * (residual ** 2)
            else:
                self.residual_var[arm][s] = 1.0  # Initial high uncertainty
# EVOLVE-BLOCK-END
