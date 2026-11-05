"""Multi-scale ensemble linear bandit with adaptive model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-scale ensemble of linear models with adaptive model selection.
    
    Each arm maintains a bank of regressors with geometrically spaced forgetting factors.
    The policy selects the arm that maximizes the reward prediction from the ensemble member
    with the lowest recent residual variance, effectively using model selection to dynamically
    switch memory horizons without explicit regime detection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Define forgetting factors: geometrically spaced from slow to fast
        # More aggressive forgetting for faster adaptation
        self.forgetting_factors = [0.995, 0.99, 0.985, 0.98, 0.975, 0.97]
        self.n_models = len(self.forgetting_factors)

        # For each arm, maintain a bank of (A, b) matrices
        # A: dim x dim, b: dim
        self.A = [[[np.eye(dim) for _ in range(self.n_models)] for _ in range(n_arms)]]
        # Actually, let's structure it as: self.A[arm][model_idx]
        self.A = [[[np.eye(dim) for _ in range(self.n_models)] for _ in range(n_arms)]]
        self.b = [[[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]]

        # Track residual variance for each (arm, model) to select best model
        # Use EMA of squared residuals
        self.residual_ema = [[[0.0 for _ in range(self.n_models)] for _ in range(n_arms)]]
        self.residual_count = [[[0 for _ in range(self.n_models)] for _ in range(n_arms)]]

        # Hyperparameters
        self.ema_alpha = 0.1  # learning rate for residual EMA
        self.exploration_weight = 0.5  # UCB exploration weight

    def _select_best_model(self, arm):
        """Select the model index with lowest recent residual variance for a given arm."""
        # Use the residual EMA to pick the best model
        # Lower residual variance means the model fits recent data better
        residuals = self.residual_ema[arm]
        # Handle case where no residuals have been observed
        if all(r == 0.0 for r in residuals):
            return 0  # Default to first model
        # Pick model with minimum residual EMA
        return int(np.argmin(residuals))

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return a fixed arm to avoid timeout
            return 0

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # For each arm, compute the best prediction using model selection
            best_score = -np.inf
            for m in range(self.n_models):
                A = self.A[a][m]
                b = self.b[a][m]

                # Compute theta = A^{-1} b
                # Use solve for numerical stability
                try:
                    theta = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    theta = np.zeros(self.d)

                # Prediction
                pred = theta @ context

                # Uncertainty: sqrt(context^T A^{-1} context)
                try:
                    A_inv_context = np.linalg.solve(A, context)
                    uncertainty = np.sqrt(max(0.0, context @ A_inv_context))
                except np.linalg.LinAlgError:
                    uncertainty = 1.0

                # UCB score
                score = pred + self.exploration_weight * uncertainty

                if score > best_score:
                    best_score = score

            scores.append(best_score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Update all models for the selected arm
        for m in range(self.n_models):
            decay = self.forgetting_factors[m]
            A = self.A[arm][m]
            b = self.b[arm][m]

            # Apply forgetting and add new observation
            A_new = decay * A + np.outer(context, context)
            b_new = decay * b + reward * context

            # Compute prediction from this model before update (for residual tracking)
            try:
                theta = np.linalg.solve(A, b)
                pred = theta @ context
                residual = reward - pred
                self.residual_ema[arm][m] = (1 - self.ema_alpha) * self.residual_ema[arm][m] + self.ema_alpha * (residual ** 2)
                self.residual_count[arm][m] += 1
            except np.linalg.LinAlgError:
                pass

            self.A[arm][m] = A_new
            self.b[arm][m] = b_new
# EVOLVE-BLOCK-END
