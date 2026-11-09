"""Non-stationary linear contextual bandit with a global context-augmented model."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model over an augmented feature space:
      x_aug = [one_hot(arm), context]
    This captures arm-specific offsets and common context effects in a single
    parameter vector, reducing the effective exploration space compared to
    independent per-arm models. Uses a forgetting factor to handle
    non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = n_arms + dim  # one-hot arm + context

        # Global ridge-regression state
        self.A = np.eye(self.d_aug) * 1.0
        self.b = np.zeros(self.d_aug)

        # Forgetting factor for non-stationarity
        self.forget = 0.995

        # Exploration parameter
        self.alpha = 1.0

    def _augment(self, context, arm):
        """Build augmented feature vector: [one_hot(arm), context]."""
        x = np.zeros(self.d_aug)
        x[arm] = 1.0
        x[self.n:] = context
        return x

    def select(self, context):
        """
        Select arm by maximizing optimistic estimate:
          E[r | a, context] + alpha * sqrt(variance)
        Since we have a single global model, we predict the reward for each
        arm using its augmented feature vector.
        """
        scores = np.zeros(self.n)
        for a in range(self.n):
            x = self._augment(context, a)
            # Predicted mean
            theta = np.linalg.solve(self.A, self.b)
            pred = theta @ x
            # Uncertainty: sqrt(x^T A^{-1} x)
            # To avoid full inverse, use solve for A^{-1} x
            Ax_inv_x = np.linalg.solve(self.A, x)
            var = x @ Ax_inv_x
            scores[a] = pred + self.alpha * np.sqrt(max(var, 0.0))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global model with a forgetting factor."""
        x = self._augment(context, arm)
        # Apply forgetting
        self.A = self.forget * self.A
        self.b = self.forget * self.b
        # Add new observation
        self.A += np.outer(x, x)
        self.b += reward * x
# EVOLVE-BLOCK-END
